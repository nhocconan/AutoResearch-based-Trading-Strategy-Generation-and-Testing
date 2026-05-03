#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation.
# Long when price breaks above Donchian upper band in bull trend (close > 1w EMA50) with volume spike.
# Short when price breaks below Donchian lower band in bear trend (close < 1w EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Donchian channels capture breakouts in trending markets.
# The 1w EMA50 filter ensures we only take breakout trades in the direction of the higher timeframe trend.
# Volume confirmation reduces false breakouts. Designed for 30-100 total trades over 4 years (7-25/year).

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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 51:  # Need at least 50 for EMA + 1 for current
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) on 1d timeframe
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 1d volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        upper_band = highest_high_20[i]
        lower_band = lowest_low_20[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Donchian breakout conditions
        breakout_long = close_val > upper_band
        breakout_short = close_val < lower_band
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_long and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_short and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price retouches middle of Donchian channel OR trend reversal
            middle_band = (upper_band + lower_band) / 2
            if close_val < middle_band or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price retouches middle of Donchian channel OR trend reversal
            middle_band = (upper_band + lower_band) / 2
            if close_val > middle_band or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals