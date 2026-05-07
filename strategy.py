# 4h_Camarilla_R3_S3_1dTrend_VolumeBreak
# Hypothesis: 4h Camarilla R3/S3 breakout with daily trend and volume confirmation
# - Uses Camarilla pivot levels (R3/S3) from daily data as strong support/resistance
# - Breakout above S3 with volume in daily uptrend = long opportunity
# - Breakdown below R3 with volume in daily downtrend = short opportunity
# - Volume spike (1.5x average) confirms institutional participation
# - Daily trend filter (EMA34) ensures alignment with higher timeframe momentum
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to Camarilla PP or volume weakens
# - Position size 0.25 targets 20-40 trades/year, avoiding fee drag
# - Camarilla levels provide structure that works across market regimes
# Timeframe: 4h, Leverage: 1.0

#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_1dTrend_VolumeBreak"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily high/low/close for Camarilla calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla pivot levels (based on previous day)
    # R4 = close + (high - low) * 1.5
    # R3 = close + (high - low) * 1.25
    # R2 = close + (high - low) * 1.166
    # R1 = close + (high - low) * 1.083
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 1.083
    # S2 = close - (high - low) * 1.166
    # S3 = close - (high - low) * 1.25
    # S4 = close - (high - low) * 1.5
    
    # Calculate for previous day (shift by 1)
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    diff = prev_high - prev_low
    r3 = prev_close + diff * 1.25
    s3 = prev_close - diff * 1.25
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 24-period average (6 periods per day in 4h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S3 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 1.5
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R3 with volume and daily downtrend
            elif close[i] < r3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below pivot or volume drops
            if close[i] < pp_aligned[i] or volume[i] < vol_ma_24[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above pivot or volume drops
            if close[i] > pp_aligned[i] or volume[i] < vol_ma_24[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals