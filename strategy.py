#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Reversal_v2
Hypothesis: On 4h timeframe, enter long when price touches Camarilla S1 (support) from above with bullish rejection (close > open) AND daily trend is up (1d close > 1d EMA34). Enter short when price touches Camarilla R1 (resistance) from below with bearish rejection (close < open) AND daily trend is down (1d close < 1d EMA34). Exit on opposite Camarilla level touch or trend reversal. Uses tight Camarilla pivot levels with price action confirmation and daily trend filter to capture reversals in both bull and bear markets. Target: 20-40 trades/year.
"""

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
    open_price = prices['open'].values
    
    # Get 1d data for Camarilla levels and daily trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for daily trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12
    s1 = prev_close_1d - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels and daily EMA to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Price action confirmation: bullish/bearish candle
        bullish_rejection = close[i] > open_price[i]  # close > open
        bearish_rejection = close[i] < open_price[i]  # close < open
        
        # Daily trend filter
        daily_uptrend = df_1d['close'].iloc[-1] > ema_34_1d[-1] if len(df_1d) > 0 else False  # Simplified: use last known
        daily_downtrend = df_1d['close'].iloc[-1] < ema_34_1d[-1] if len(df_1d) > 0 else False
        
        # More robust daily trend using aligned value (approximation for current bar)
        # Since we can't use future data, use the EMA value from the last completed 1d bar
        # We'll use a simplified approach: compare 1d close to its EMA using available aligned data
        # For now, we use the aligned EMA value as proxy for trend direction
        if i >= len(prices):  # safety
            continue
            
        # Actually, we need to compute daily trend properly without look-ahead
        # Let's recompute: for each 4h bar, we need to know if the DAILY trend is up/down
        # based on completed 1d bars only
        # We'll compute this outside the loop for efficiency
        
        # Recompute properly: calculate daily trend alignment
        pass  # We'll implement correctly below
    
    # Re-implementing with proper daily trend calculation
    # Calculate if each 1d bar is above/below its EMA34
    daily_trend_up = (df_1d['close'].values > ema_34_1d)  # True if 1d close > EMA34
    daily_trend_down = (df_1d['close'].values < ema_34_1d)  # True if 1d close < EMA34
    
    # Align daily trend to 4h timeframe (wait for completed 1d bar)
    daily_trend_up_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_up.astype(float))
    daily_trend_down_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(daily_trend_up_aligned[i]) or np.isnan(daily_trend_down_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Price action confirmation: bullish/bearish candle
        bullish_rejection = close[i] > open_price[i]  # close > open
        bearish_rejection = close[i] < open_price[i]  # close < open
        
        # Touch conditions: price touches Camarilla level
        # Long: touches S1 from above (low <= S1 and close > S1) with bullish rejection
        touches_s1 = low[i] <= s1_aligned[i] and close[i] > s1_aligned[i]
        # Short: touches R1 from below (high >= R1 and close < R1) with bearish rejection
        touches_r1 = high[i] >= r1_aligned[i] and close[i] < r1_aligned[i]
        
        if position == 0:
            # Long: touches S1 + bullish rejection + daily uptrend
            long_signal = touches_s1 and bullish_rejection and daily_trend_up_aligned[i] > 0.5
            
            # Short: touches R1 + bearish rejection + daily downtrend
            short_signal = touches_r1 and bearish_rejection and daily_trend_down_aligned[i] > 0.5
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: touches R1 (opposite level) OR daily trend turns down
            if touches_r1 or daily_trend_down_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: touches S1 (opposite level) OR daily trend turns up
            if touches_s1 or daily_trend_up_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_Reversal_v2"
timeframe = "4h"
leverage = 1.0