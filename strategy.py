# 4h_donchian_breakout_12h_trend_volume_v1
# Hypothesis: On 4h timeframe, breakout of Donchian(20) channels with 12h trend filter and volume confirmation.
# Long when price closes above upper Donchian band with volume > 1.3x average and 12h trend up.
# Short when price closes below lower Donchian band with volume > 1.3x average and 12h trend down.
# Exit when price touches the opposite Donchian band or volume drops below average.
# Uses tight entry conditions to limit trades to ~20-50 per year, avoiding fee drag.
# Works in both bull and bear markets via trend filter and volatility-based channels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian band or volume drops below average
            if close[i] <= low_roll[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian band or volume drops below average
            if close[i] >= high_roll[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.3x average volume
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            # 12h trend filter
            trend_up = close[i] > ema_50_12h_aligned[i]
            trend_down = close[i] < ema_50_12h_aligned[i]
            
            # Long entry: price closes above upper Donchian band with volume and uptrend
            if close[i] > high_roll[i] and volume_ok and trend_up:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below lower Donchian band with volume and downtrend
            elif close[i] < low_roll[i] and volume_ok and trend_down:
                position = -1
                signals[i] = -0.25
    
    return signals