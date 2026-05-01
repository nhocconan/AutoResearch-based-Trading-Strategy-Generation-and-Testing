#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and ATR volatility filter.
# Long when price breaks above Donchian upper band AND 1d EMA50 rising AND ATR(14) > 0.5 * ATR(50) (volatility expansion).
# Short when price breaks below Donchian lower band AND 1d EMA50 falling AND ATR(14) > 0.5 * ATR(50).
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 12h timeframe to capture medium-term trends with volatility confirmation.
# Donchian channels provide clear breakout levels that work in both trending and volatile markets.
# 1d EMA50 trend filter ensures alignment with higher timeframe momentum.
# ATR volatility filter ensures we only trade during periods of sufficient market movement, reducing false breakouts in chop.

name = "12h_Donchian20_1dEMA50_ATR_VolFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d EMA50 slope (rising/falling)
    ema_50_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    ema_50_rising = ema_50_slope > 0
    ema_50_falling = ema_50_slope < 0
    
    # Calculate ATR(14) and ATR(50) for volatility filter on 12h timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / np.where(atr_50 == 0, np.nan, atr_50)  # Avoid division by zero
    vol_expansion = atr_ratio > 0.5  # ATR(14) > 0.5 * ATR(50)
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for Donchian, ATR and EMA calculations
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND 1d EMA50 rising AND volatility expansion
            if (curr_high > donchian_high[i] and 
                ema_50_rising[i] and 
                vol_expansion[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND 1d EMA50 falling AND volatility expansion
            elif (curr_low < donchian_low[i] and 
                  ema_50_falling[i] and 
                  vol_expansion[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR 1d EMA50 falls (trend change)
            if (curr_low < donchian_low[i] or 
                ema_50_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR 1d EMA50 rises (trend change)
            if (curr_high > donchian_high[i] or 
                ema_50_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals