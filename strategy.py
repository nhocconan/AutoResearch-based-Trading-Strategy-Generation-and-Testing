#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Chop Regime Filter
# Long when: Jaw < Teeth < Lips (bullish alignment) AND price > Lips AND 1d volume spike AND chop < 61.8 (trending)
# Short when: Jaw > Teeth > Lips (bearish alignment) AND price < Lips AND 1d volume spike AND chop < 61.8 (trending)
# Uses Williams Alligator (SMAs with offsets) for trend, 1d volume spike for confirmation, chop regime to avoid ranging markets
# Designed for 12h timeframe to capture medium-term trends with low trade frequency (~20-50 trades/year)
# Works in bull (trend continuation) and bear (trend continuation) markets by following Alligator alignment

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume spike and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume spike (2.0x 20-bar MA)
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike_1d = vol_1d > (2.0 * vol_ma_20)
    else:
        volume_spike_1d = np.zeros(len(df_1d), dtype=bool)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 1d chop regime (EHLERS CHOPPINESS INDEX)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    chop_length = 14
    if len(high_1d) >= chop_length:
        # True range
        tr1 = np.abs(high_1d - low_1d)
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr_sum = pd.Series(tr).rolling(window=chop_length, min_periods=chop_length).sum().values
        
        # ATR
        atr = pd.Series(tr).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
        atr_sum = pd.Series(atr).rolling(window=chop_length, min_periods=chop_length).sum().values
        
        # Chop = 100 * log10(atr_sum / tr_sum) / log10(chop_length)
        with np.errstate(divide='ignore', invalid='ignore'):
            chop = 100 * (np.log10(atr_sum) - np.log10(tr_sum)) / np.log10(chop_length)
        chop = np.nan_to_num(chop, nan=100.0)
    else:
        chop = np.full(len(df_1d), 100.0)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, offset 8 bars
    # Teeth: 8-period SMMA, offset 5 bars
    # Lips: 5-period SMMA, offset 3 bars
    def smma(source, length):
        if len(source) < length:
            return np.full_like(source, np.nan)
        sma = pd.Series(source).rolling(window=length, min_periods=length).mean().values
        smma_vals = np.full_like(source, np.nan)
        smma_vals[length-1] = sma[length-1]
        for i in range(length, len(source)):
            smma_vals[i] = (smma_vals[i-1] * (length-1) + source[i]) / length
        return smma_vals
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply offsets (shift right by offset bars)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set first offset bars to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Jaw < Teeth < Lips (bullish) AND price > Lips AND volume spike AND chop < 61.8 (trending)
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and 
                close[i] > lips[i] and 
                volume_spike_1d_aligned[i] and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish) AND price < Lips AND volume spike AND chop < 61.8 (trending)
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  close[i] < lips[i] and 
                  volume_spike_1d_aligned[i] and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (Jaw > Teeth OR Teeth > Lips) OR price < Lips
            if (jaw[i] > teeth[i] or teeth[i] > lips[i] or close[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (Jaw < Teeth OR Teeth < Lips) OR price > Lips
            if (jaw[i] < teeth[i] or teeth[i] < lips[i] or close[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals