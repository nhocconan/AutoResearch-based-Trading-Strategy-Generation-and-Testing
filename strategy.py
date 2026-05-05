#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high AND 1w close > 1w EMA50 AND volume > 2x 20 EMA
# Short when price breaks below 20-period Donchian low AND 1w close < 1w EMA50 AND volume > 2x 20 EMA
# Uses discrete sizing (0.25) to limit fee drag. Target: 7-25 trades/year per symbol.
# Donchian provides strong trend-following structure; EMA50 filters higher-timeframe trend; volume spike confirms momentum.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Uses 1w for HTF trend to avoid counter-trend trades and 1d for Donchian breakout timing.

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian levels (20-period) based on previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian levels: based on previous 20 bars
    # Upper = max(high_1d over last 20 periods)
    # Lower = min(low_1d over last 20 periods)
    # We calculate rolling max/min then shift by 1 to use previous bar's levels for current bar breakout
    high_rolling_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift to use previous bar's levels (breakout of previous bar's Donchian)
    donchian_high = np.roll(high_rolling_max, 1)
    donchian_low = np.roll(low_rolling_min, 1)
    donchian_high[0] = np.nan  # First value invalid after roll
    donchian_low[0] = np.nan
    
    # Align Donchian levels to prices timeframe (1d to 1d is identity but keeps pattern consistent)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1w = close_1w > ema_50_1w
    downtrend_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian high AND 1w uptrend AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                uptrend_1w_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian low AND 1w downtrend AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  downtrend_1w_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian low OR 1w trend changes to downtrend
            if (close[i] < donchian_low_aligned[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian high OR 1w trend changes to uptrend
            if (close[i] > donchian_high_aligned[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals