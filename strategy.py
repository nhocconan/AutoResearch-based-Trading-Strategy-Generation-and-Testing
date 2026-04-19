# 12h_4hEMA34_Trend_Filter_With_1d_Volume_Spike
# Hypothesis: Use 1d volume spike as entry trigger with 4h EMA34 trend filter on 12h timeframe.
# Enter long when: 4h EMA34 trending up AND 1d volume > 2x 20-day average AND price > 4h EMA34.
# Enter short when: 4h EMA34 trending down AND 1d volume > 2x 20-day average AND price < 4h EMA34.
# Exit when trend reverses. Volume spike ensures momentum, EMA34 filters whipsaw.
# Designed for 5-15 trades/year per symbol to avoid fee drag.
# Works in bull/bear: volume spikes occur in both regimes, trend filter avoids counter-trend trades.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_4hEMA34_Trend_Filter_With_1d_Volume_Spike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA34 on 4h
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1d data for volume spike
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 to be ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_4h_aligned[i]
        vol_spike = vol_spike_1d_aligned[i] > 0.5  # Convert back to boolean
        
        # Determine trend direction from EMA34 slope (using 3-period change)
        if i >= 37:  # Need 3 periods for slope
            ema34_prev = ema34_4h_aligned[i-3]
            ema34_slope = ema34 - ema34_prev
            is_uptrend = ema34_slope > 0
            is_downtrend = ema34_slope < 0
        else:
            is_uptrend = False
            is_downtrend = False
        
        if position == 0:
            # Look for volume spike with trend alignment
            if vol_spike:
                if is_uptrend and price > ema34:
                    signals[i] = 0.25
                    position = 1
                elif is_downtrend and price < ema34:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: trend turns down or price crosses below EMA34
            if is_downtrend or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend turns up or price crosses above EMA34
            if is_uptrend or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals