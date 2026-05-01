#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Chop Regime Filter
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via SMAs with future shifts.
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment).
# Volume spike confirms institutional participation. Chop regime (CHOP>61.8) avoids ranging markets.
# Designed for 12h timeframe to capture medium-term trends with minimal fee drag.
# Target: 12-30 trades/year to stay within HARD MAX of 200 total over 4 years.

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopFilter_v1"
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
    
    # 1d HTF data for volume spike and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Volume confirmation: current 1d volume > 2.0 * 20-period average volume
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (vol_ma_20 * 2.0)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Chop regime: CHOP(14) > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_14 = np.zeros(len(close_1d))
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_high_low = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_high_low / (atr_14 * 14)) / np.log10(2)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    # SMMA = smoothed moving average (similar to EMA with alpha=1/period)
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Close) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    smma_5 = smma(close, 5)
    smma_8 = smma(close, 8)
    smma_13 = smma(close, 13)
    
    # Alligator lines with shifts (future shift removed for no look-ahead)
    # Lips: 5-period SMMA shifted 3 bars back -> use current value
    lips = smma_5  # Already effectively shifted by calculation method
    # Teeth: 8-period SMMA shifted 5 bars back
    teeth = np.full_like(smma_8, np.nan)
    teeth[5:] = smma_8[:-5]  # Shift 5 bars back
    # Jaw: 13-period SMMA shifted 8 bars back
    jaw = np.full_like(smma_13, np.nan)
    jaw[8:] = smma_13[:-8]  # Shift 8 bars back
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need sufficient history for SMMA and shifts
    
    for i in range(start_idx, n):
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator trend condition
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Regime filter: avoid choppy markets (CHOP > 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        # Volume confirmation
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish alignment, volume spike, trending regime
            if bullish_alignment and vol_spike and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, volume spike, trending regime
            elif bearish_alignment and vol_spike and trending_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish alignment or chop regime
            if bearish_alignment or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish alignment or chop regime
            if bullish_alignment or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals