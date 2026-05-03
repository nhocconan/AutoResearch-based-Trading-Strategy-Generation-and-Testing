#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses actual Donchian channel calculation from prior 1w bar to derive upper/lower bands.
# Long when price breaks above upper band with volume > 1.5x 20-period MA and close > 1w EMA50 (uptrend).
# Short when price breaks below lower band with volume spike and close < 1w EMA50 (downtrend).
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Donchian channels provide structural support/resistance; 1w EMA50 filters counter-trend trades.
# Volume confirmation reduces false breakouts. Works in bull/bear via trend alignment.

name = "12h_Donchian20_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian calculation and EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) from prior 1w bar (need completed 1w bar only)
    # Upper band = max(high over past 20 completed 1w bars)
    # Lower band = min(low over past 20 completed 1w bars)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Shift by 1 to use completed 1w bar only (no look-ahead)
    high_1w_shifted = np.roll(high_1w, 1)
    low_1w_shifted = np.roll(low_1w, 1)
    high_1w_shifted[0] = np.nan  # First value invalid after shift
    low_1w_shifted[0] = np.nan
    
    # Calculate rolling max/min on 1w timeframe
    high_max_20 = pd.Series(high_1w_shifted).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w_shifted).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    
    # Volume regime: current 12h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        upper_band = donchian_upper_aligned[i]
        lower_band = donchian_lower_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: break above upper band with volume spike in uptrend
            if close_val > upper_band and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume spike in downtrend
            elif close_val < lower_band and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower band (reversal) OR trend turns down
            if close_val < lower_band or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper band (reversal) OR trend turns up
            if close_val > upper_band or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals