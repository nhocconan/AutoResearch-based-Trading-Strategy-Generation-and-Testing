#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla Pivot Level Touch with 1-day Volume Confirmation and 1-day Trend Filter
# Long when price touches S1/S2 pivot levels + daily EMA(34) uptrend + daily volume spike
# Short when price touches R1/R2 pivot levels + daily EMA(34) downtrend + daily volume spike
# Camarilla levels provide clear support/resistance from prior day's range
# Volume spike confirms institutional participation in the reversal/trend
# Trend filter ensures we trade with higher timeframe momentum
# Targets 20-50 total trades per year to minimize fee drag

name = "4h_Camarilla_Pivot_Touch_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for pivot levels, trend filter, and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily volume average for volume confirmation
    daily_volume = df_1d['volume'].values
    vol_avg_1d = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Camarilla pivot levels from previous day's range
    # Standard Camarilla: H4 = C + ((H-L) * 1.5), L4 = C - ((H-L) * 1.5)
    # We use S1, S2 for longs and R1, R2 for shorts
    # S1 = C - ((H-L) * 1.125), S2 = C - ((H-L) * 1.25)
    # R1 = C + ((H-L) * 1.125), R2 = C + ((H-L) * 1.25)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    s1 = prev_close - ((prev_high - prev_low) * 1.125)
    s2 = prev_close - ((prev_high - prev_low) * 1.25)
    r1 = prev_close + ((prev_high - prev_low) * 1.125)
    r2 = prev_close + ((prev_high - prev_low) * 1.25)
    
    # Align pivot levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        vol_avg_val = vol_avg_1d_aligned[i]
        s1_val = s1_aligned[i]
        s2_val = s2_aligned[i]
        r1_val = r1_aligned[i]
        r2_val = r2_aligned[i]
        
        # Volume spike: current daily volume > 1.5 * 20-day average
        daily_vol_idx = i // 96  # 96 4h bars in a day (24*60/15)
        # We need to get the actual daily volume value for today
        # Since we're on 4h timeframe, we check if current 4h bar's time corresponds to a new day
        # Instead, we use the aligned daily volume average which is already mapped to each 4h bar
        vol_spike = volume[i] > (1.5 * vol_avg_val) if not np.isnan(vol_avg_val) else False
        
        if position == 0:
            # Enter long: price touches S1 or S2 + daily uptrend + volume spike
            if ((low[i] <= s1_val and close[i] > s1_val) or 
                (low[i] <= s2_val and close[i] > s2_val)) and \
               close[i] > ema34_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price touches R1 or R2 + daily downtrend + volume spike
            elif ((high[i] >= r1_val and close[i] < r1_val) or 
                  (high[i] >= r2_val and close[i] < r2_val)) and \
                 close[i] < ema34_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches R1 or daily trend turns down
            if high[i] >= r1_val or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches S1 or daily trend turns up
            if low[i] <= s1_val or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals