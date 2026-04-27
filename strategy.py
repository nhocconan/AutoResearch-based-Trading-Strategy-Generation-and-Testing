# 1d_Camarilla_Pivot_Reversal_WeeklyTrend_v1
# Hypothesis: Camarilla pivot reversals on daily timeframe with weekly trend filter and volume spike.
# Works in bull/bear by fading extremes with trend alignment and volume confirmation.
# Target: 20-50 trades/year (80-200 total over 4 years) to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day (based on previous day's range)
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    d_close = df_1d['close'].values
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    
    # Previous day's values for today's levels
    prev_close = np.roll(d_close, 1)
    prev_high = np.roll(d_high, 1)
    prev_low = np.roll(d_low, 1)
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else d_close[0]
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else d_high[0]
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else d_low[0]
    
    # Calculate Camarilla levels
    R4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    R2 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    R1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    S1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    S2 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    S4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align daily levels to 15-minute bars (assuming 15m input, but works for any)
    R4_15m = align_htf_to_ltf(prices, df_1d, R4)
    R3_15m = align_htf_to_ltf(prices, df_1d, R3)
    R2_15m = align_htf_to_ltf(prices, df_1d, R2)
    R1_15m = align_htf_to_ltf(prices, df_1d, R1)
    S1_15m = align_htf_to_ltf(prices, df_1d, S1)
    S2_15m = align_htf_to_ltf(prices, df_1d, S2)
    S3_15m = align_htf_to_ltf(prices, df_1d, S3)
    S4_15m = align_htf_to_ltf(prices, df_1d, S4)
    
    # Weekly trend filter (EMA34 on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for calculations
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_15m[i]) or np.isnan(S1_15m[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema34_1w_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long reversal at S1 with weekly uptrend and volume spike
            if close[i] <= S1_15m[i] and weekly_trend > 0 and vol_spike_val:
                signals[i] = size
                position = 1
            # Short reversal at R1 with weekly downtrend and volume spike
            elif close[i] >= R1_15m[i] and weekly_trend < 0 and vol_spike_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R1 or weekly trend turns down
            if close[i] >= R1_15m[i] or weekly_trend <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches S1 or weekly trend turns up
            if close[i] <= S1_15m[i] or weekly_trend >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_Pivot_Reversal_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0