#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d Volume Spike + Chop Regime Filter
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend presence and direction.
# Combined with 1d volume spike for institutional confirmation and chop filter to avoid ranging markets.
# Designed to capture strong trends in both bull and bear markets while minimizing false signals.
# Target: 20-40 trades/year to minimize fee drag while maintaining edge.

name = "4h_WilliamsAlligator_1dVolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for volume spike calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume spike: current 1d volume > 2.0 * 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_20_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Chop regime filter: CHOP(14) > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) for chop calculation
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over last 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(ATR14)/ (HH14-LL14)) / log10(14)
    chop = np.zeros_like(close)
    for i in range(14, n):
        if atr_14[i] > 0 and (hh_14[i] - ll_14[i]) > 0:
            chop[i] = 100 * np.log10(atr_14[i] * 14) / np.log10(hh_14[i] - ll_14[i])
        else:
            chop[i] = 50  # Neutral when undefined
    
    chop_filter = chop < 50  # Favor trending markets (chop < 50)
    
    # Williams Alligator: SMAs of median price with offsets
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 30  # Need sufficient history for Alligator and other indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions: 
        # Uptrend: Lips > Teeth > Jaw (all aligned and pointing up)
        # Downtrend: Lips < Teeth < Jaw (all aligned and pointing down)
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        uptrend = lips_val > teeth_val and teeth_val > jaw_val
        downtrend = lips_val < teeth_val and teeth_val < jaw_val
        
        # Volume spike and chop filter
        vol_spike = vol_spike_1d_aligned[i] > 0.5  # Convert back to boolean
        trend_regime = chop_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend, volume spike, trending regime
            if uptrend and vol_spike and trend_regime:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend, volume spike, trending regime
            elif downtrend and vol_spike and trend_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Alligator trend reversal (Lips < Teeth)
            if lips_val < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Alligator trend reversal (Lips > Teeth)
            if lips_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals