#!/usr/bin/env python3
"""
6h_400_500_DMA_Crossover_Pullback_v1
Hypothesis: Uses 400 and 500-period exponential moving averages on 1h timeframe to identify long-term trend,
with 6h price retracement to the 400 EMA as entry signal. Works in bull markets by buying dips in uptrends
and in bear markets by selling rallies in downtrends. Target: 15-25 trades/year to minimize fee drag.
"""

name = "6h_400_500_DMA_Crossover_Pullback_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 600:
        return np.zeros(n)
    
    # Get 1h data for EMA calculation (trend identification)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 500:
        return np.zeros(n)
    
    # Calculate 400 and 500 EMA on 1h close
    close_1h = df_1h['close'].values
    ema_400 = pd.Series(close_1h).ewm(span=400, adjust=False, min_periods=400).mean().values
    ema_500 = pd.Series(close_1h).ewm(span=500, adjust=False, min_periods=500).mean().values
    
    # Determine trend: 400 EMA > 500 EMA = uptrend, < = downtrend
    trend_up = ema_400 > ema_500
    
    # Align trend signals to 6h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1h, trend_up)
    
    # 6h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 600
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(trend_up_aligned[i]):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        is_uptrend = trend_up_aligned[i]
        is_downtrend = not is_uptrend
        
        # Entry conditions: price retracement to 400 EMA on 6b chart
        # Need to calculate 400 EMA on 6h timeframe for entry reference
        # But we only have 1h EMA aligned - so we'll use price action relative to recent swings
        
        # Alternative: use 6h price crossing 400 EMA (from 1h) as dynamic support/resistance
        # Since we don't have 6h EMA, we'll use a simpler approach: 
        # In uptrend: buy when price pulls back to near recent low and shows rejection
        # In downtrend: sell when price rallies to near recent high and shows rejection
        
        # Calculate 20-period high/low on 6h for context
        if i >= 20:
            recent_high = np.max(high[i-20:i])
            recent_low = np.min(low[i-20:i])
        else:
            recent_high = high[i]
            recent_low = low[i]
        
        # Define proximity zones (within 1% of recent extremes)
        proximity_threshold = 0.01
        near_high = abs(close[i] - recent_high) / recent_high < proximity_threshold
        near_low = abs(close[i] - recent_low) / recent_low < proximity_threshold
        
        # Rejection signals: wick rejection from levels
        body_size = abs(close[i] - prices['open'].iloc[i])
        upper_wick = high[i] - max(close[i], prices['open'].iloc[i])
        lower_wick = min(close[i], prices['open'].iloc[i]) - low[i]
        
        # Strong rejection: wick at least 2x body size
        strong_lower_rejection = lower_wick > 2 * body_size and body_size > 0
        strong_upper_rejection = upper_wick > 2 * body_size and body_size > 0
        
        if position == 0:
            # In uptrend: look for long near support (recent low) with rejection
            if is_uptrend and near_low and strong_lower_rejection:
                signals[i] = 0.25
                position = 1
            # In downtrend: look for short near resistance (recent high) with rejection
            elif is_downtrend and near_high and strong_upper_rejection:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite rejection
            if position == 1:  # Long position
                # Exit if trend turns down or we get rejection at resistance
                if not is_uptrend or (near_high and strong_upper_rejection):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit if trend turns up or we get rejection at support
                if is_uptrend or (near_low and strong_lower_rejection):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals