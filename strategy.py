#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day Donchian high AND 1w close > 1w EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below 20-day Donchian low AND 1w close < 1w EMA50 AND volume > 2.0x 20-period average
# Exit when price crosses 1w EMA50 (trend reversal)
# Uses 1d primary timeframe with 1w HTF for trend filter and Donchian structure
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) based on proven Donchian breakout performance
# Donchian channels provide robust price channels; 1w EMA50 filters for higher-timeframe trend; volume confirms breakout validity
# Works in both bull and bear markets by following the 1w trend while using 1d for entry timing

name = "1d_Donchian20_Breakout_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for trend filter and Donchian levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian levels on 1w data (based on previous bar's OHLC)
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    
    donchian_high = pd.Series(h_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(l_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (wait for 1w bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND 1w close > 1w EMA50 AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND 1w close < 1w EMA50 AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA50 (trend reversal)
            if close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w EMA50 (trend reversal)
            if close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals