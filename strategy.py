#!/usr/bin/env python3
# Hypothesis: 4h 123-Reversal with volume confirmation and 1-day trend filter.
# The 123-reversal pattern identifies short-term exhaustion: 
#   - For longs: price makes a new low, then a higher low, then breaks above the pullback high.
#   - For shorts: price makes a new high, then a lower high, then breaks below the pullback low.
# This pattern works in both trending and ranging markets as it captures mean-reversion within the trend.
# We use 1-day EMA as a trend filter: only take longs when price > daily EMA50, shorts when price < daily EMA50.
# Volume confirmation: require volume > 1.5x average volume to avoid false breakouts.
# Position size: 0.25 to balance risk and return.
# Target: 80-180 total trades over 4 years (20-45/year).

name = "4h_123_Reversal_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Average volume for confirmation (20-period)
    avg_vol = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for pattern detection
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(avg_vol[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for 123-reversal long setup
            # Condition 1: new low (lower than previous low)
            new_low = low[i] < low[i-1]
            # Condition 2: higher low (current low > previous low)
            higher_low = low[i] > low[i-1]
            # Actually: we need to find a sequence: low, then higher low, then breakout
            # Simplified: look for higher low after a decline, then break above recent high
            
            # Alternative simpler 123: 
            # Point 1: recent swing low
            # Point 2: pullback high that is lower than previous high
            # Point 3: break above point 2
            
            # We'll use: 
            # - Point 1: lowest low in last 5 bars
            # - Point 2: highest high after that low but before current bar
            # - Point 3: current close breaks above point 2
            
            # Find lowest low in lookback period
            lookback = 10
            if i >= lookback:
                lowest_low_idx = np.argmin(low[i-lookback:i])
                point1_low = low[i-lookback + lowest_low_idx]
                
                # Find highest high after that point but before current bar
                if lowest_low_idx < lookback - 1:
                    high_after_low = high[i-lookback + lowest_low_idx + 1:i]
                    if len(high_after_low) > 0:
                        point2_high = np.max(high_after_low)
                        # Breakout condition: close > point2_high
                        breakout_long = close[i] > point2_high
                        
                        # Additional conditions:
                        # - Point 2 high should be less than the high before point 1 (showing weakness)
                        if lowest_low_idx >= 2:
                            high_before_low = high[i-lookback:i-lookback + lowest_low_idx]
                            if len(high_before_low) > 0:
                                point1_high = np.max(high_before_low)
                                # Weakness condition: point2_high < point1_high
                                weakness = point2_high < point1_high
                            else:
                                weakness = True
                        else:
                            weakness = True
                        
                        # Volume confirmation
                        vol_confirm = volume[i] > 1.5 * avg_vol[i]
                        
                        # Trend filter: price above daily EMA50 for longs
                        trend_filter = close[i] > ema_50_1d_aligned[i]
                        
                        if breakout_long and weakness and vol_confirm and trend_filter:
                            signals[i] = 0.25
                            position = 1
                            continue
            
            # Check for 123-reversal short setup
            if i >= lookback:
                highest_high_idx = np.argmax(high[i-lookback:i])
                point1_high = high[i-lookback + highest_high_idx]
                
                # Find lowest low after that point but before current bar
                if highest_high_idx < lookback - 1:
                    low_after_high = low[i-lookback + highest_high_idx + 1:i]
                    if len(low_after_high) > 0:
                        point2_low = np.min(low_after_high)
                        # Breakdown condition: close < point2_low
                        breakdown_short = close[i] < point2_low
                        
                        # Additional conditions:
                        # - Point 2 low should be greater than the low before point 1 (showing weakness)
                        if highest_high_idx >= 2:
                            low_before_high = low[i-lookback:i-lookback + highest_high_idx]
                            if len(low_before_high) > 0:
                                point1_low = np.min(low_before_high)
                                # Weakness condition: point2_low > point1_low
                                weakness = point2_low > point1_low
                            else:
                                weakness = True
                        else:
                            weakness = True
                        
                        # Volume confirmation
                        vol_confirm = volume[i] > 1.5 * avg_vol[i]
                        
                        # Trend filter: price below daily EMA50 for shorts
                        trend_filter = close[i] < ema_50_1d_aligned[i]
                        
                        if breakdown_short and weakness and vol_confirm and trend_filter:
                            signals[i] = -0.25
                            position = -1
                            continue
        
        elif position == 1:
            # Exit long: price drops below the entry signal level or trend changes
            # Exit when: close drops below the low of the signal bar, or trend filter fails
            if close[i] < low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above the entry signal level or trend changes
            # Exit when: close rises above the high of the signal bar, or trend filter fails
            if close[i] > high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals