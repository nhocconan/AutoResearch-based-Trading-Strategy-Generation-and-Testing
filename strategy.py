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
    
    # === 1d VWAP (Volume Weighted Average Price) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = np.divide(vwap_numerator, vwap_denominator, 
                        out=np.full_like(vwap_numerator, np.nan), 
                        where=vwap_denominator!=0)
    
    # === 1d VWAP Standard Deviation (20 periods) ===
    # Calculate deviation from VWAP
    dev_squared = (typical_price_1d - vwap_1d) ** 2
    # Weighted variance using volume
    vwap_var = np.divide(np.cumsum(dev_squared * volume_1d), 
                         vwap_denominator,
                         out=np.full_like(vwap_numerator, np.nan),
                         where=vwap_denominator!=0)
    vwap_std = np.sqrt(np.maximum(vwap_var, 0))
    
    # VWAP Bands (2 standard deviations)
    vwap_upper = vwap_1d + 2.0 * vwap_std
    vwap_lower = vwap_1d - 2.0 * vwap_std
    
    # Align to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vwap_upper_aligned = align_htf_to_ltf(prices, df_1d, vwap_upper)
    vwap_lower_aligned = align_htf_to_ltf(prices, df_1d, vwap_lower)
    
    # === 1d Volume Spike Detection ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1d ADX (14 periods) for trend strength ===
    # Calculate directional movement
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - high_1d[i-1]), 
                    abs(low_1d[i] - low_1d[i-1]))
    
    # Wilder's smoothing (alpha = 1/period)
    atr_14 = np.zeros(len(tr))
    plus_di_14 = np.zeros(len(plus_dm))
    minus_di_14 = np.zeros(len(minus_dm))
    
    # Initialize first values
    atr_14[0] = tr[0]
    plus_di_14[0] = plus_dm[0]
    minus_di_14[0] = minus_dm[0]
    
    # Smooth subsequent values
    for i in range(1, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
        plus_di_14[i] = (plus_di_14[i-1] * 13 + plus_dm[i]) / 14
        minus_di_14[i] = (minus_di_14[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate DX
    dx = np.zeros(len(atr_14))
    mask = (plus_di_14 + minus_di_14) > 0
    dx[mask] = 100 * np.abs(plus_di_14[mask] - minus_di_14[mask]) / (plus_di_14[mask] + minus_di_14[mask])
    
    # Calculate ADX (smoothed DX)
    adx_14 = np.zeros(len(dx))
    if len(dx) > 0:
        adx_14[0] = dx[0]
        for i in range(1, len(dx)):
            adx_14[i] = (adx_14[i-1] * 13 + dx[i]) / 14
    
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for calculations
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(vwap_upper_aligned[i]) or 
            np.isnan(vwap_lower_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 2.0
        
        # ADX filter: only trade when trend is strong (ADX > 25)
        adx_strong = adx_14_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price crosses above VWAP upper band with volume confirmation and strong trend
            if close[i] > vwap_upper_aligned[i] and vol_confirm and adx_strong:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price crosses below VWAP lower band with volume confirmation and strong trend
            elif close[i] < vwap_lower_aligned[i] and vol_confirm and adx_strong:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price returns to VWAP
        elif position == 1:
            # Exit long: price crosses below VWAP
            if close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above VWAP
            if close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_VWAP_2Std_Dev_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0