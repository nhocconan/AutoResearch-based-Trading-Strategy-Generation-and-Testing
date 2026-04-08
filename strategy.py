# 6h_renko_brick_trend_volume
# Hypothesis: Renko brick trend with volume confirmation on 6h timeframe.
# Uses 1% brick size to filter noise. Long when green brick forms with volume > 1.5x average.
# Short when red brick forms with volume > 1.5x average.
# Renko bricks reduce whipsaw in sideways markets while capturing trends.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_renko_brick_trend_volume"
timeframe = "6h"
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
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Renko bricks (1% brick size)
    brick_size = 0.01  # 1% of price
    renko_direction = np.zeros(n)  # 1=up brick, -1=down brick, 0=no new brick
    brick_low = np.full(n, np.nan)
    brick_high = np.full(n, np.nan)
    
    # Initialize first brick
    if n > 0:
        brick_low[0] = close[0] - (close[0] % brick_size)
        brick_high[0] = brick_low[0] + brick_size
        renko_direction[0] = 1  # Start with up brick assumption
    
    # Calculate Renko bricks
    for i in range(1, n):
        if close[i] >= brick_high[i-1]:
            # Upward brick(s)
            bricks = int((close[i] - brick_high[i-1]) // brick_size) + 1
            renko_direction[i] = 1
            brick_low[i] = brick_high[i-1]
            brick_high[i] = brick_low[i] + bricks * brick_size
        elif close[i] <= brick_low[i-1]:
            # Downward brick(s)
            bricks = int((brick_low[i-1] - close[i]) // brick_size) + 1
            renko_direction[i] = -1
            brick_high[i] = brick_low[i-1]
            brick_low[i] = brick_high[i] - bricks * brick_size
        else:
            # Inside brick, no new brick
            renko_direction[i] = 0
            brick_low[i] = brick_low[i-1]
            brick_high[i] = brick_high[i-1]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: red brick forms (trend reversal)
            if renko_direction[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: green brick forms (trend reversal)
            if renko_direction[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Renko brick entries: green brick (long) and red brick (short)
            if renko_direction[i] == 1 and volume_ok:
                position = 1
                signals[i] = 0.25
            elif renko_direction[i] == -1 and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals