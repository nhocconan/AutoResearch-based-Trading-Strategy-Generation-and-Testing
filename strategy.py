#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: Daily Donchian(20) breakout with 1w EMA trend filter and volume confirmation
    # Long: price breaks above upper Donchian band AND weekly EMA(50) rising AND volume > 2x average
    # Short: price breaks below lower Donchian band AND weekly EMA(50) falling AND volume > 2x average
    # Exit: price touches opposite Donchian band OR weekly EMA flips direction
    # Using 1d timeframe for low trade frequency (target 7-25/year), Donchian for structure,
    # Weekly EMA for major trend filter, volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly EMA slope (rising/falling)
    ema_slope = np.full_like(ema_50_1w_aligned, np.nan)
    for i in range(1, len(ema_slope)):
        if not np.isnan(ema_50_1w_aligned[i]) and not np.isnan(ema_50_1w_aligned[i-1]):
            ema_slope[i] = ema_50_1w_aligned[i] - ema_50_1w_aligned[i-1]
    
    # Calculate daily Donchian channels (20-period)
    upper_donchian = np.full(n, np.nan)
    lower_donchian = np.full(n, np.nan)
    for i in range(20, n):
        upper_donchian[i] = np.max(high[i-20:i])
        lower_donchian[i] = np.min(low[i-20:i])
    
    # Get daily volume for confirmation (>2x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(ema_slope[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Weekly EMA trend filter
        ema_rising = ema_slope[i] > 0
        ema_falling = ema_slope[i] < 0
        
        # Donchian breakout conditions
        breakout_upper = close[i] > upper_donchian[i]
        breakout_lower = close[i] < lower_donchian[i]
        
        # Exit conditions: touch opposite band OR weekly EMA flips direction
        exit_long = close[i] < lower_donchian[i] or (position == 1 and ema_falling)
        exit_short = close[i] > upper_donchian[i] or (position == -1 and ema_rising)
        
        # Entry logic: Donchian breakout + weekly EMA trend + volume confirmation
        long_entry = breakout_upper and ema_rising and volume_spike[i]
        short_entry = breakout_lower and ema_falling and volume_spike[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_breakout_ema_volume_v1"
timeframe = "1d"
leverage = 1.0