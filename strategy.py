#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (EMA50) and 1d volume spike confirmation.
# Uses Camarilla pivot levels (R1/S1) from prior 1h for structure, EMA50 on 4h for trend direction,
# and volume spike on 1d (>1.5x 20-bar EMA-scaled avg volume) for conviction.
# Discrete position sizing (0.0, ±0.20) minimizes fee churn. Session filter (08-20 UTC) reduces noise.
# Designed to capture intraday breakouts aligned with higher timeframe trend and volume conviction.
# Targets 15-30 trades/year per symbol (~60-120 over 4 years) to avoid fee drag.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # --- 1h Indicators (LTF) ---
    # ATR(14) for volatility (used in Camarilla calc)
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # EMA50 on 4h for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # ATR-scaled volume MA: 20-period EMA of volume / ATR
    high_shift_1d = np.roll(high_1d, 1)
    low_shift_1d = np.roll(low_1d, 1)
    close_shift_1d = np.roll(close_1d, 1)
    high_shift_1d[0] = high_1d[0]
    low_shift_1d[0] = low_1d[0]
    close_shift_1d[0] = close_1d[0]
    
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - close_shift_1d), np.abs(low_1d - close_shift_1d)))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    vol_atr_ratio_1d = volume_1d / (atr_14_1d + 1e-10)
    vol_atr_ema_20_1d = pd.Series(vol_atr_ratio_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = vol_atr_ratio_1d > (1.5 * vol_atr_ema_20_1d)
    
    # Camarilla levels (R1, S1) from prior 1h bar
    camarilla_range_1h = high - low
    r1_1h = close + 1.1 * camarilla_range_1h / 12.0
    s1_1h = close - 1.1 * camarilla_range_1h / 12.0
    
    # Align Camarilla levels to current 1h (no delay needed as they're based on completed prior bar)
    r1_1h_aligned = r1_1h  # already based on prior completed 1h bar
    s1_1h_aligned = s1_1h  # already based on prior completed 1h bar
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if outside session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_spike_1d[i]) or
            np.isnan(r1_1h_aligned[i]) or
            np.isnan(s1_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R1 AND uptrend on 4h AND volume spike on 1d
            if close[i] > r1_1h_aligned[i] and uptrend and volume_spike_1d[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 AND downtrend on 4h AND volume spike on 1d
            elif close[i] < s1_1h_aligned[i] and downtrend and volume_spike_1d[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 (mean reversion) OR trend breaks
            if close[i] < s1_1h_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 (mean reversion) OR trend breaks
            if close[i] > r1_1h_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals