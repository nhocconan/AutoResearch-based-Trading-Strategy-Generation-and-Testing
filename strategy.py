#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Daily high/low/close for Camarilla calculation (previous day's values)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels based on previous day's range
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    s1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    pp = (daily_high + daily_low + daily_close) / 3  # Pivot point
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2-period average (24h of 12h bars)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 2)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(vol_ma_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_2[i] * 1.5
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and daily downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below pivot or volume drops
            if close[i] < pp_aligned[i] or volume[i] < vol_ma_2[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above pivot or volume drops
            if close[i] > pp_aligned[i] or volume[i] < vol_ma_2[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R1/S1 breakout with daily trend and volume confirmation
# - Camarilla R1/S1 from previous day's range act as intraday support/resistance
# - Breakout above S1 with volume in daily uptrend = long opportunity
# - Breakdown below R1 with volume in daily downtrend = short opportunity
# - Volume spike (1.5x average) confirms participation
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to daily pivot (PP) or volume weakens significantly
# - Position size 0.25 targets ~25-40 trades/year, avoiding fee drag
# - Daily trend filter ensures alignment with higher timeframe momentum
# - Uses actual Camarilla formula from institutional trading, not improvised levels