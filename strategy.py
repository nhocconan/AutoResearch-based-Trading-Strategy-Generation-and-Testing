# 6h_Camarilla_R3S3_Breakout_1dTrend_Volume_1wTrend
# Uses Camarilla pivot levels from 1d with R3/S3 as breakout levels, confirmed by 1d trend and volume spike
# Exit when price returns to daily pivot (PP) or trend reverses
# Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# Target: 15-35 trades/year to avoid fee drag
# Uses 1w trend filter to avoid counter-trend trades in strong weekly trends

#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume_1wTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Formula uses previous day's high, low, close
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Camarilla levels
    pp = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r3 = prev_close + range_hl * 1.1
    s3 = prev_close - range_hl * 1.1
    r4 = prev_close + range_hl * 1.5
    s4 = prev_close - range_hl * 1.5
    
    # Align Camarilla levels to 6h timeframe (they update daily at 00:00 UTC)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly trend filter: EMA(21) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume spike detection: 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 24)  # Wait for EMA(34) and volume MA(24)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume and aligned daily/weekly uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            daily_uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            weekly_uptrend = ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]
            
            if close[i] > r3_aligned[i] and vol_condition and daily_uptrend and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and aligned daily/weekly downtrend
            elif close[i] < s3_aligned[i] and vol_condition and not daily_uptrend and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to PP or trend reverses
            if close[i] < pp_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to PP or trend reverses
            if close[i] > pp_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with multi-timeframe trend alignment
# - Camarilla R3/S3 act as breakout levels (beyond normal R1/S1)
# - Requires volume spike (2x 4-day average) to confirm institutional participation
# - Daily EMA(34) trend filter ensures trading with intra-day momentum
# - Weekly EMA(21) filter prevents counter-trend trades in strong weekly trends
# - Long: break above R3 in daily/weekly uptrend with volume
# - Short: break below S3 in daily/weekly downtrend with volume
# - Exit when price returns to daily pivot (PP) or daily trend reverses
# - Position size 0.25 targets 15-35 trades/year to minimize fee drag
# - Works in bull (buy R3 breaks in uptrends) and bear (sell S3 breaks in downtrends)