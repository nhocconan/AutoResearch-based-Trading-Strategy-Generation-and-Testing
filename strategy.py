#!/usr/bin/env python3
"""
4h Volume-Weighted RSI Mean Reversion + 1d ATR Regime Filter
Hypothesis: In ranging markets, extreme RSI values revert to mean. Volume-weighted RSI reduces false signals.
ATR regime filter ensures we only trade when volatility is elevated enough for meaningful moves.
Works in both bull and bear markets by fading momentum exhaustion spikes.
Target: 20-40 trades/year (80-160 over 4 years) to minimize fee drag.
"""

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ATR(14) for regime filter
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume-weighted RSI(14) on 4h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Volume-weighted gain/loss
    vol_ratio = volume / pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    vol_gain = gain * vol_ratio
    vol_loss = loss * vol_ratio
    
    avg_gain = pd.Series(vol_gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(vol_loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    vw_rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 14, 14)  # volume MA, VW RSI, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vw_rsi[i]) or np.isnan(avg_gain[i]) or np.isnan(avg_loss[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1d ATR is above its 50-period median (elevated volatility)
        if i >= 50:
            atr_median = np.nanmedian(atr_1d_aligned[max(0, i-50):i+1])
            high_vol_regime = atr_1d_aligned[i] > atr_median
        else:
            high_vol_regime = False
        
        if position == 0:
            # Look for entry signals in high volatility regime only
            if high_vol_regime:
                # Long: VW RSI < 25 (oversold) with volume confirmation
                long_entry = (vw_rsi[i] < 25) and (volume[i] > np.nanmedian(volume[max(0, i-20):i+1]))
                # Short: VW RSI > 75 (overbought) with volume confirmation
                short_entry = (vw_rsi[i] > 75) and (volume[i] > np.nanmedian(volume[max(0, i-20):i+1]))
                
                if long_entry:
                    signals[i] = 0.25
                    position = 1
                elif short_entry:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: VW RSI > 50 (mean reversion) OR VW RSI > 65 (early exit)
            if vw_rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: VW RSI < 50 (mean reversion) OR VW RSI < 35 (early exit)
            if vw_rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolumeWeightedRSI_MeanReversion_1dATR_Regime"
timeframe = "4h"
leverage = 1.0