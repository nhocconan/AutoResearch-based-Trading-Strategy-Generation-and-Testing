#%%
#!/usr/bin/env python3
# 1d_donchian_breakout_1w_trend_volume_v1
# Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
# Captures strong directional moves while avoiding chop. Works in bull/bear by following weekly trend.
# Target: 20-30 trades/year via tight entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Get weekly data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: daily volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < daily Donchian low OR price < weekly EMA50
            if (close[i] < low_20[i]) or (close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > daily Donchian high OR price > weekly EMA50
            if (close[i] > high_20[i]) or (close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > daily Donchian high + volume + price > weekly EMA50
            if (close[i] > high_20[i]) and volume_filter[i] and (close[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < daily Donchian low + volume + price < weekly EMA50
            elif (close[i] < low_20[i]) and volume_filter[i] and (close[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
#%%