#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla levels from daily pivot provide institutional support/resistance.
# Breakout above R3 or below S3 with volume spike indicates strong momentum.
# 1d EMA34 filter ensures alignment with higher timeframe trend to avoid false breakouts.
# Works in bull/bear markets by trading breakouts in direction of daily trend.
# Target: 12-30 trades/year (50-120 total over 4 years).

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # P = (H + L + C) / 3
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    daily_pivot = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    daily_range = prev_day_high - prev_day_low
    
    r1 = prev_day_close + daily_range * 1.1 / 12
    s1 = prev_day_close - daily_range * 1.1 / 12
    r2 = prev_day_close + daily_range * 1.1 / 6
    s2 = prev_day_close - daily_range * 1.1 / 6
    r3 = prev_day_close + daily_range * 1.1 / 4
    s3 = prev_day_close - daily_range * 1.1 / 4
    r4 = prev_day_close + daily_range * 1.1 / 2
    s4 = prev_day_close - daily_range * 1.1 / 2
    
    # Align daily Camarilla levels to 6h timeframe (completed daily bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
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
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > R3 with volume confirmation and above daily EMA34
            if curr_close > curr_r3 and curr_volume_confirm and curr_close > curr_ema_34:
                signals[i] = 0.25
                position = 1
            # Short breakout: price < S3 with volume confirmation and below daily EMA34
            elif curr_close < curr_s3 and curr_volume_confirm and curr_close < curr_ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below R1 (failed breakout) or reverses below daily EMA34
            if curr_close < r1_aligned[i] or curr_close < curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above S1 (failed breakdown) or reverses above daily EMA34
            if curr_close > s1_aligned[i] or curr_close > curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals