# 1. Hypothesis: 4-hour Donchian breakout with 1-day ATR volatility filter and 1-week trend filter
# Long when price breaks above 4h Donchian(20) high AND 1d ATR volatility is low (trending regime) AND 1w EMA(50) is rising
# Short when price breaks below 4h Donchian(20) low AND 1d ATR volatility is low AND 1w EMA(50) is falling
# Exit when price reverses back into the Donchian channel
# Designed for 15-30 trades/year to minimize fee drag while capturing strong trends in any market regime
# Uses volatility filter to avoid false breakouts in choppy markets and trend filter to align with higher timeframe momentum

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1-day data for ATR and 1-week data for EMA - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-day ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR calculation
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    atr_values = atr.values
    
    # Calculate 1-week EMA (50-period) for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_values = ema_50.values
    
    # Align HTF indicators to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_values)
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_values)
    
    # Calculate 4-hour Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    highest_high_values = highest_high.values
    lowest_low_values = lowest_low.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(ema_aligned[i]) or 
            np.isnan(highest_high_values[i]) or np.isnan(lowest_low_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_val = atr_aligned[i]
        ema_val = ema_aligned[i]
        donchian_high = highest_high_values[i]
        donchian_low = lowest_low_values[i]
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + low volatility (ATR < 1.5 * 20-period ATR mean) + rising weekly EMA
            if (price > donchian_high and 
                atr_val < 1.5 * np.nanmean(atr_aligned[max(0, i-20):i]) and
                ema_val > ema_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + low volatility + falling weekly EMA
            elif (price < donchian_low and 
                  atr_val < 1.5 * np.nanmean(atr_aligned[max(0, i-20):i]) and
                  ema_val < ema_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price reverses back into the Donchian channel
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses back below Donchian high
                if price < donchian_high:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses back above Donchian low
                if price > donchian_low:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_ATR_VolatilityFilter_1wEMA_Trend"
timeframe = "4h"
leverage = 1.0