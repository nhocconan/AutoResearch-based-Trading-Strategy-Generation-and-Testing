# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike_v3
# Hypothesis: Camarilla pivot breakout (R1/S1) with daily trend filter and volume spike
# works in bull/bear by capturing breakouts with trend alignment and filtering false breakouts
# via volume confirmation. Target 20-40 trades/year to avoid fee drag.
# Uses 1d trend (EMA34) and volume spike (>2x 20-period average) for confirmation.
# Entry: Close breaks R1 (long) or S1 (short) with 1d trend alignment and volume spike.
# Exit: Close crosses back through Camarilla pivot point (PP).
# Position size: 0.25 to limit drawdown.

#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike_v3"
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
    
    # Calculate Camarilla levels from previous day
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # We need previous day's H, L, C - using daily data
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    # Use previous day's data to avoid look-ahead
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.concatenate([[np.nan], daily_high[:-1]])
    prev_low = np.concatenate([[np.nan], daily_low[:-1]])
    prev_close = np.concatenate([[np.nan], daily_close[:-1]])
    
    # Camarilla calculations
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    
    # Align daily levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d trend filter: EMA(34) on daily close
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.0 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above R1 with 1d uptrend and volume spike
            if close[i] > r1_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with 1d downtrend and volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below pivot point (PP)
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above pivot point (PP)
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals