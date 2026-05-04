#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume Weighted Average Price (VWAP) deviation with 1d ATR regime filter and volume spike confirmation
# In ranging markets (low ATR regime), price tends to revert to VWAP. In trending markets (high ATR),
# we breakout in the direction of the trend. Volume spike (>1.5x 20 EMA) confirms institutional participation.
# Uses discrete sizing 0.25 to limit risk. Target: 80-180 trades over 4 years (20-45/year).
# Works in bull/bear: ATR regime adapts to market conditions, VWAP provides dynamic mean/reversion anchor.

name = "6h_VWAP_1dATR_Regime_VolumeSpike"
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
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime detection
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 6h timeframe (completed 1d bar only)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(cum_vol != 0, cum_pv / cum_vol, typical_price)
    
    # Calculate 6h ATR(10) for entry threshold
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr_6h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        # Price deviation from VWAP in ATR units
        price_dev_atr = abs(close[i] - vwap[i]) / atr_6h[i]
        
        if position == 0:
            # Regime-based entry logic
            if atr_1d_aligned[i] < np.nanmedian(atr_1d_aligned[max(0, i-100):i]):  # Low ATR = ranging
                # Mean reversion: price deviates significantly from VWAP
                if price_dev_atr > 2.0 and volume_confirm:
                    if close[i] < vwap[i]:  # Below VWAP -> long
                        signals[i] = 0.25
                        position = 1
                    else:  # Above VWAP -> short
                        signals[i] = -0.25
                        position = -1
            else:  # High ATR = trending
                # Breakout: price breaks VWAP with volume
                if price_dev_atr > 1.5 and volume_confirm:
                    if close[i] > vwap[i]:  # Above VWAP -> long
                        signals[i] = 0.25
                        position = 1
                    else:  # Below VWAP -> short
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: price returns to VWAP OR volatility collapses OR volume drops
            if (abs(close[i] - vwap[i]) < 0.5 * atr_6h[i] or 
                atr_1d_aligned[i] < np.nanmedian(atr_1d_aligned[max(0, i-50):i]) * 0.5 or
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to VWAP OR volatility collapses OR volume drops
            if (abs(close[i] - vwap[i]) < 0.5 * atr_6h[i] or 
                atr_1d_aligned[i] < np.nanmedian(atr_1d_aligned[max(0, i-50):i]) * 0.5 or
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals