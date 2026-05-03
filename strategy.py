#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long: Close breaks above upper Donchian(20) AND price > 1w EMA50 (uptrend) AND volume > 2.0x 20-period MA
# Short: Close breaks below lower Donchian(20) AND price < 1w EMA50 (downtrend) AND volume > 2.0x 20-period MA
# Exit: Opposite Donchian breakout or EMA50 trend reversal.
# Discrete sizing 0.25. Target: 30-100 total trades over 4 years (7-25/year).
# Donchian channels provide strong trend-following structure; 1w EMA50 filters higher timeframe trend;
# volume confirmation reduces false signals. Works in bull via long signals with trend alignment
# and in bear via short signals with trend alignment.

name = "1d_Donchian20_1wEMA50_Volume"
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
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels using previous 20 periods (standard formula)
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    # Using rolling window on 1d data
    roll_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    roll_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels (already on 1d, no alignment needed but keep for consistency)
    upper_channel = roll_high
    lower_channel = roll_low
    
    # Volume regime: current 1d volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above upper Donchian AND uptrend AND volume spike
            if close_val > upper_channel[i] and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Donchian AND downtrend AND volume spike
            elif close_val < lower_channel[i] and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below lower Donchian OR trend turns down
            if close_val < lower_channel[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above upper Donchian OR trend turns up
            if close_val > upper_channel[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals