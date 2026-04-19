#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1dCMO_VolumeBreakout_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for CMO and volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily CMO (Chande Momentum Oscillator) 14-period
    # CMO = (Sum of gains - Sum of losses) / (Sum of gains + Sum of losses) * 100
    delta = np.diff(close_1d, prepend=close_1d[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    sum_gains = pd.Series(gains).rolling(window=14, min_periods=14).sum().values
    sum_losses = pd.Series(losses).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    denominator = sum_gains + sum_losses
    cmo_1d = np.where(denominator != 0, (sum_gains - sum_losses) / denominator * 100, 0)
    cmo_1d_aligned = align_htf_to_ltf(prices, df_1d, cmo_1d)
    
    # Calculate daily ATR for volatility regime filter
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volatility regime: ATR ratio (current ATR / 50-period ATR mean)
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(cmo_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        cmo = cmo_1d_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        atr = atr_1d_aligned[i]
        
        volume_confirmed = vol > 1.8 * vol_ma
        # Volatility regime: prefer moderate volatility (avoid extreme low/high vol)
        vol_regime_ok = (atr_ratio_val > 0.6) & (atr_ratio_val < 2.2)
        
        if position == 0:
            # Long: CMO oversold (< -30) turning up + volume + vol regime
            if cmo < -30 and cmo > cmo_1d_aligned[i-1] and volume_confirmed and vol_regime_ok:
                signals[i] = 0.25
                position = 1
            # Short: CMO overbought (> 30) turning down + volume + vol regime
            elif cmo > 30 and cmo < cmo_1d_aligned[i-1] and volume_confirmed and vol_regime_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: CMO crosses above 30 (overbought) or volatility too high
            if cmo > 30 or atr_ratio_val > 2.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: CMO crosses below -30 (oversold) or volatility too high
            if cmo < -30 or atr_ratio_val > 2.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals