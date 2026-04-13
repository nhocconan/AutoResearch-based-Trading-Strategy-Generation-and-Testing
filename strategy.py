#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
    # Long: price breaks above upper Donchian(20) AND 12h EMA50 > price (uptrend) AND volume > 1.3x 20-bar avg
    # Short: price breaks below lower Donchian(20) AND 12h EMA50 < price (downtrend) AND volume > 1.3x 20-bar avg
    # Exit: price touches opposite Donchian band or middle (20-bar avg)
    # Using 4h timeframe for optimal trade frequency (target 19-50/year), Donchian for structure,
    # 12h EMA50 for multi-timeframe trend alignment, and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    upper_donchian = np.full(n, np.nan)
    lower_donchian = np.full(n, np.nan)
    middle_donchian = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_donchian[i] = np.max(high[i-20:i])
        lower_donchian[i] = np.min(low[i-20:i])
        middle_donchian[i] = (upper_donchian[i] + lower_donchian[i]) / 2
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: >1.3x 20-bar average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if data not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or np.isnan(middle_donchian[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_upper = close[i] > upper_donchian[i]
        breakout_lower = close[i] < lower_donchian[i]
        
        # Exit conditions: touch opposite band or middle
        exit_long = close[i] < middle_donchian[i]
        exit_short = close[i] > middle_donchian[i]
        
        # Trend filter: 12h EMA50 direction
        uptrend = ema_50_aligned[i] > close[i]  # Price below EMA50 = uptrend (EMA acts as support)
        downtrend = ema_50_aligned[i] < close[i]  # Price above EMA50 = downtrend (EMA acts as resistance)
        
        # Entry logic: Donchian breakout + trend alignment + volume confirmation
        long_entry = breakout_upper and uptrend and volume_spike[i]
        short_entry = breakout_lower and downtrend and volume_spike[i]
        
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

name = "4h_12h_donchian_breakout_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0