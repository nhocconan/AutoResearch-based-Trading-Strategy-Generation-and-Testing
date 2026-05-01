#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 1-2-3 Reversal Pattern with 1d Trend Filter and Volume Confirmation.
# Uses classic 1-2-3 reversal (Ross Hook) for high-probability trend continuation entries.
# 1d EMA50 establishes higher timeframe trend direction to avoid counter-trend trades.
# Volume confirmation ensures institutional participation. Target: 15-25 trades/year.
# Works in bull markets (long 1-2-3 at pullbacks) and bear markets (short 1-2-3 at rallies).
# The 1-2-3 pattern has proven edge in capturing institutional order flow after retracements.

name = "6h_123Reversal_1dTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track 1-2-3 pattern points
    point1 = np.full(n, np.nan)  # extreme point
    point2 = np.full(n, np.nan)  # retracement point
    point3 = np.full(n, np.nan)  # test of point1 level
    pattern_stage = np.zeros(n)  # 0: none, 1: point1 found, 2: point2 found, 3: point3 found
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema_50 = ema_50_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Skip if data not ready
        if (np.isnan(curr_ema_50) or np.isnan(curr_vol_ma)):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        volume_ok = curr_volume > curr_vol_ma
        
        # Update 1-2-3 pattern tracking
        if i > start_idx:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            # Check for new swing high/low (point1 candidates)
            # Swing high: current high > previous high and next candle's high (we use previous as proxy)
            # Actually look for: recent swing points
            
            # Simpler approach: track recent extremes
            if i >= 5:
                lookback = 5
                recent_high = np.max(high[i-lookback:i+1])
                recent_low = np.min(low[i-lookback:i+1])
                
                # New swing high made
                if curr_high == recent_high and curr_high > np.max(high[i-lookback:i]):
                    # Potential point1 for downtrend (recent high)
                    point1[i] = curr_high
                    point2[i] = np.nan
                    point3[i] = np.nan
                    pattern_stage[i] = 1
                
                # New swing low made
                elif curr_low == recent_low and curr_low < np.min(low[i-lookback:i]):
                    # Potential point1 for uptrend (recent low)
                    point1[i] = curr_low
                    point2[i] = np.nan
                    point3[i] = np.nan
                    pattern_stage[i] = 1
                else:
                    # Inherit previous pattern stage
                    point1[i] = point1[i-1]
                    point2[i] = point2[i-1]
                    point3[i] = point3[i-1]
                    pattern_stage[i] = pattern_stage[i-1]
                    
                    # Update point2 (retracement) if we have point1
                    if pattern_stage[i-1] == 1 and not np.isnan(point1[i-1]):
                        # For uptrend pattern (point1 is low): point2 is retracement high
                        if curr_high > point1[i-1] and curr_high < point1[i-1] * 1.05:  # reasonable retracement
                            point2[i] = curr_high
                            pattern_stage[i] = 2
                        # For downtrend pattern (point1 is high): point2 is retracement low
                        elif curr_low < point1[i-1] and curr_low > point1[i-1] * 0.95:
                            point2[i] = curr_low
                            pattern_stage[i] = 2
                    
                    # Update point3 (test of point1) if we have point2
                    elif pattern_stage[i-1] == 2 and not np.isnan(point2[i-1]):
                        # For uptrend: point3 should test point1 low but not break it
                        if curr_low <= point1[i-1] * 1.005 and curr_low >= point1[i-1] * 0.995:  # within 0.5%
                            point3[i] = curr_low
                            pattern_stage[i] = 3
                        # For downtrend: point3 should test point1 high but not break it
                        elif curr_high >= point1[i-1] * 0.995 and curr_high <= point1[i-1] * 1.005:
                            point3[i] = curr_high
                            pattern_stage[i] = 3
        
        # Inherit values if not set in this iteration
        if i == start_idx:
            point1[i] = np.nan
            point2[i] = np.nan
            point3[i] = np.nan
            pattern_stage[i] = 0
        elif np.isnan(point1[i]):
            point1[i] = point1[i-1]
            point2[i] = point2[i-1]
            point3[i] = point3[i-1]
            pattern_stage[i] = pattern_stage[i-1]
        
        curr_p1 = point1[i]
        curr_p2 = point2[i]
        curr_p3 = point3[i]
        curr_stage = pattern_stage[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long 1-2-3: point1=low, point2=retracement high, point3=test of low
            # Conditions: uptrend on 1d, point3 formed, volume confirmation
            if (not np.isnan(curr_p1) and not np.isnan(curr_p2) and not np.isnan(curr_p3) and
                curr_stage == 3 and
                curr_close > curr_ema_50 and  # 1d uptrend filter
                curr_close > curr_p1 and      # price above point1 (breakout)
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short 1-2-3: point1=high, point2=retracement low, point3=test of high
            # Conditions: downtrend on 1d, point3 formed, volume confirmation
            elif (not np.isnan(curr_p1) and not np.isnan(curr_p2) and not np.isnan(curr_p3) and
                  curr_stage == 3 and
                  curr_close < curr_ema_50 and  # 1d downtrend filter
                  curr_close < curr_p1 and      # price below point1 (breakdown)
                  volume_ok):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below point2 (retracement high) or trend change
            if (not np.isnan(curr_p2) and curr_close < curr_p2) or curr_close < curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above point2 (retracement low) or trend change
            if (not np.isnan(curr_p2) and curr_close > curr_p2) or curr_close > curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals