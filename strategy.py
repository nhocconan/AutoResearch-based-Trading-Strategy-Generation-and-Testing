#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when price breaks above 20-day high AND 1w EMA50 is rising AND volume > 1.5x 20-day MA.
# Short when price breaks below 20-day low AND 1w EMA50 is falling AND volume > 1.5x 20-day MA.
# Uses discrete sizing 0.30 to balance return and drawdown. Target: 30-100 total trades over 4 years.
# Works in bull via breakout longs and bear via breakdown shorts when aligned with weekly trend.

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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w EMA50 slope (rising/falling)
    ema_slope = np.diff(ema_50_1w_aligned, prepend=ema_50_1w_aligned[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 1d volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: Breakout above 20-day high AND weekly EMA rising AND volume spike
            if close_val > highest_20[i] and ema_rising[i] and vol_spike:
                signals[i] = 0.30
                position = 1
            # Short: Breakdown below 20-day low AND weekly EMA falling AND volume spike
            elif close_val < lowest_20[i] and ema_falling[i] and vol_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Breakdown below 20-day low OR weekly EMA starts falling OR volume drops
            if close_val < lowest_20[i] or not ema_rising[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Breakout above 20-day high OR weekly EMA starts rising OR volume drops
            if close_val > highest_20[i] or ema_rising[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals