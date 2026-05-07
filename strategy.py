#!/usr/bin/env python3
name = "4h_1d_Camarilla_R3_S3_Trend_Volume"
timeframe = "4h"
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
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily Camarilla R3/S3 from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels (R3 and S3)
    s3 = prev_close - (range_hl * 1.125)  # S3: close - range*1.125
    r3 = prev_close + (range_hl * 1.125)  # R3: close + range*1.125
    
    # Align daily levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 6)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S3 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_6[i] * 2.0  # Strong volume spike
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R3 with volume and daily downtrend
            elif close[i] < r3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S3 or volume drops significantly
            if close[i] < s3_aligned[i] or volume[i] < vol_ma_6[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R3 or volume drops significantly
            if close[i] > r3_aligned[i] or volume[i] < vol_ma_6[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with daily trend and volume confirmation
# - Uses stronger R3/S3 levels (vs S1/R1) for fewer, higher-quality trades
# - Breakout above S3 with volume in daily uptrend = long opportunity
# - Breakdown below R3 with volume in daily downtrend = short opportunity
# - Volume spike (2.0x average) confirms strong institutional participation
# - Daily EMA(34) trend filter ensures alignment with higher timeframe momentum
# - Tight entry conditions target ~20-40 trades/year to minimize fee drag
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend) markets
# - Exit when price returns to S3/R3 or volume weakens significantly
# - Position size 0.25 balances return potential with drawdown control
# - Designed to avoid overtrading while maintaining edge in BTC/ETH markets