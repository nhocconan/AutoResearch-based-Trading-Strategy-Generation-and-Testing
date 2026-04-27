# 101120
# 4h_Donchian20_Breakout_VolumeSpike_TrendFilter_1d
# Hypothesis: Donchian(20) breakout on 4h with 1d EMA50 trend filter and volume spike (2x 20-period average) captures
# strong momentum moves while avoiding false breakouts. Works in bull (breakouts up) and bear (breakouts down) regimes.
# Volume confirmation ensures breakouts have conviction. Trend filter avoids counter-trend trades.
# Target: 20-40 trades/year (80-160 total over 4 years) to stay under 400 total trade limit.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA for trend filter (1d)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50 = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 2x 20-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50[i]) or np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above upper Donchian + trend up + volume spike
        long_breakout = (close[i] > highest_high[i-1] and close[i] > ema50[i] and volume_spike[i])
        # Short conditions: price breaks below lower Donchian + trend down + volume spike
        short_breakout = (close[i] < lowest_low[i-1] and close[i] < ema50[i] and volume_spike[i])
        
        if long_breakout:
            signals[i] = 0.30
            position = 1
        elif short_breakout:
            signals[i] = -0.30
            position = -1
        # Exit conditions: opposite Donchian breakout with volume
        elif position == 1 and close[i] < lowest_low[i-1] and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high[i-1] and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_TrendFilter_1d"
timeframe = "4h"
leverage = 1.0