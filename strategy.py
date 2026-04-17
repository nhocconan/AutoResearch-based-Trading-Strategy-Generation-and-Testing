#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Use 1d EMA(34) for trend direction, 1d ATR(14) for volatility filter,
# and 4h volume regime filter. Enter only during high-volume sessions (08-20 UTC)
# when price is beyond 1.5x ATR from EMA in trend direction. Exit when price
# returns to within 0.5x ATR of EMA. This captures trending moves while avoiding
# chop. Target: 15-35 trades/year with controlled risk.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 1d EMA (34-period) for trend direction ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Wilder's EMA (alpha = 1/period)
    alpha = 1.0 / 34
    ema_34 = np.full_like(close_1d, np.nan)
    if len(close_1d) > 0:
        ema_34[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_34[i] = alpha * close_1d[i] + (1 - alpha) * ema_34[i-1]
    
    # === 1d ATR (14-period) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing for ATR
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align 1d indicators to 1h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === 4h Volume regime filter ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # 20-period EMA of volume on 4h
    alpha_vol = 2.0 / (20 + 1)
    vol_ema_20 = np.full_like(volume_4h, np.nan)
    if len(volume_4h) > 0:
        vol_ema_20[0] = volume_4h[0]
        for i in range(1, len(volume_4h)):
            vol_ema_20[i] = alpha_vol * volume_4h[i] + (1 - alpha_vol) * vol_ema_20[i-1]
    
    # Volume regime: high when current > 1.5x EMA
    vol_regime_4h = volume_4h > vol_ema_20 * 1.5
    vol_regime_aligned = align_htf_to_ltf(prices, df_4h, vol_regime_4h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: ensure indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_regime_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Distance from EMA in ATR units
        if atr_14_aligned[i] <= 0:
            dist_atr = 0
        else:
            dist_atr = (close[i] - ema_34_aligned[i]) / atr_14_aligned[i]
        
        # Entry: price beyond 1.5 ATR from EMA in trend direction
        if position == 0:
            if dist_atr > 1.5:  # Strong uptrend
                signals[i] = 0.20
                position = 1
                continue
            elif dist_atr < -1.5:  # Strong downtrend
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit: price returns to within 0.5 ATR of EMA (mean reversion)
        elif position == 1:
            if dist_atr < 0.5:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            if dist_atr > -0.5:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA34_ATR_VolumeRegime_Distance_v1"
timeframe = "1h"
leverage = 1.0