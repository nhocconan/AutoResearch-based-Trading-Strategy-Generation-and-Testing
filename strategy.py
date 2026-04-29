#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla levels provide institutional support/resistance; breakouts capture momentum
# 1d EMA34 filters for higher timeframe trend alignment; volume confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (fades at extremes in range)
# Target: 20-50 trades/year (80-200 total) to stay within fee drag limits

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Based on previous day's OHLC: R4, R3, R2, R1, PP, S1, S2, S3, S4
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    prev_day_open = df_1d['open'].shift(1).values
    
    # Camarilla equations
    camarilla_pp = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    camarilla_range = prev_day_high - prev_day_low
    r1 = camarilla_pp + (camarilla_range * 1.1 / 12)
    r2 = camarilla_pp + (camarilla_range * 1.1 / 6)
    r3 = camarilla_pp + (camarilla_range * 1.1 / 4)
    r4 = camarilla_pp + (camarilla_range * 1.1 / 2)
    s1 = camarilla_pp - (camarilla_range * 1.1 / 12)
    s2 = camarilla_pp - (camarilla_range * 1.1 / 6)
    s3 = camarilla_pp - (camarilla_range * 1.1 / 4)
    s4 = camarilla_pp - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (completed daily bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # warmup for 1d EMA, Camarilla, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish breakout above R3 with volume and trend alignment
            if curr_close > curr_r3 and curr_volume_confirm and curr_close > curr_ema_34:
                signals[i] = 0.25
                position = 1
            # Bearish breakdown below S3 with volume and trend alignment
            elif curr_close < curr_s3 and curr_volume_confirm and curr_close < curr_ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when price returns to Camarilla PP level or breaks below S3
            camarilla_pp = (df_1d['high'].shift(1).iloc[i-1] + df_1d['low'].shift(1).iloc[i-1] + df_1d['close'].shift(1).iloc[i-1]) / 3.0
            camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
            if curr_close < camarilla_pp_aligned[i] or curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when price returns to Camarilla PP level or breaks above R3
            camarilla_pp = (df_1d['high'].shift(1).iloc[i-1] + df_1d['low'].shift(1).iloc[i-1] + df_1d['close'].shift(1).iloc[i-1]) / 3.0
            camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
            if curr_close > camarilla_pp_aligned[i] or curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals