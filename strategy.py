#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA (34-period) for trend direction ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA with proper Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / 34
    ema_34 = np.full_like(close_1d, np.nan)
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
    
    # Align all indicators to 1h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === 4h Volume profile for regime filter ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period average volume on 4h timeframe
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    # Volume regime: high volume when current > 1.5x average
    vol_regime_4h = volume_4h > vol_ma_20 * 1.5
    vol_regime_aligned = align_htf_to_ltf(prices, df_4h, vol_regime_4h.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 1h volume > 1.3x 20-period average
        vol_ma_20_1h = np.zeros_like(volume)
        for j in range(len(volume)):
            if j >= 19:
                vol_ma_20_1h[j] = np.mean(volume[j-19:j+1])
            else:
                vol_ma_20_1h[j] = np.mean(volume[max(0, j-9):j+1]) if j > 0 else volume[0]
        vol_confirm = volume[i] > vol_ma_20_1h[i] * 1.3
        
        # Entry logic: only enter when flat AND in high volume regime (4h)
        if position == 0:
            # Long: price above EMA34 + volatility filter + volume confirmation + high vol regime
            if (close[i] > ema_34_aligned[i] and 
                atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                vol_confirm and
                vol_regime_aligned[i] > 0.5):  # high volume regime
                signals[i] = 0.20
                position = 1
                continue
            # Short: price below EMA34 + volatility filter + volume confirmation + high vol regime
            elif (close[i] < ema_34_aligned[i] and 
                  atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                  vol_confirm and
                  vol_regime_aligned[i] > 0.5):  # high volume regime
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA34_ATR_Volume_Regime_Filter_v1"
timeframe = "1h"
leverage = 1.0