#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (EMA50) and 1d volume spike confirmation.
# Uses Camarilla pivot levels (R1/S1) from prior 1h for intraday structure, 4h EMA50 for trend direction,
# and 1d ATR-normalized volume spike (>1.8x 20-bar ATR-scaled avg volume) for conviction.
# Discrete position sizing (0.0, ±0.20) minimizes fee churn. Session filter (08-20 UTC) reduces noise.
# Designed to capture intraday breakouts aligned with higher timeframe trend and volume confirmation.
# Targets 15-35 trades/year per symbol (60-140 total over 4 years).

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_1dATRVolumeSpike_v1"
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
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # --- 1h Indicators (LTF) ---
    # ATR(14) for volatility
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Camarilla levels (R1, S1) from prior 1h bar
    camarilla_range = high - low
    r1_1h = close + 1.1 * camarilla_range / 4.0
    s1_1h = close - 1.1 * camarilla_range / 4.0
    
    # Align to current 1h (wait for completed 1h bar)
    r1_1h_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), r1_1h)
    s1_1h_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), s1_1h)
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # EMA(50) for trend direction
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
    
    # ATR(14) for 1d volatility normalization
    high_shift_1d = np.roll(high_1d, 1)
    low_shift_1d = np.roll(low_1d, 1)
    close_shift_1d = np.roll(close_1d, 1)
    high_shift_1d[0] = high_1d[0]
    low_shift_1d[0] = low_1d[0]
    close_shift_1d[0] = close_1d[0]
    
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - close_shift_1d), np.abs(low_1d - close_shift_1d)))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-scaled volume MA: 20-period average of volume / ATR
    vol_atr_ratio_1d = volume_1d / (atr_14_1d + 1e-10)
    vol_atr_ma_20_1d = pd.Series(vol_atr_ratio_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_atr_ratio_1d > (1.8 * vol_atr_ma_20_1d)
    
    # Align volume spike to 1h (wait for completed 1d bar)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(r1_1h_aligned[i]) or
            np.isnan(s1_1h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R1 AND volume spike AND uptrend on 4h
            if close[i] > r1_1h_aligned[i] and volume_spike_1d_aligned[i] and uptrend:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 AND volume spike AND downtrend on 4h
            elif close[i] < s1_1h_aligned[i] and volume_spike_1d_aligned[i] and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 (mean reversion)
            if close[i] < s1_1h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 (mean reversion)
            if close[i] > r1_1h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals