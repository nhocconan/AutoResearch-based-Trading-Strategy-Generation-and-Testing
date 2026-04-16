#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume spike and 1d chop regime filter.
# Long when Williams %R(14) < -80 (oversold) AND volume > 1.5x 20-period 1d average AND chop > 61.8 (ranging market).
# Short when Williams %R(14) > -20 (overbought) AND volume > 1.5x 20-period 1d average AND chop > 61.8.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position size 0.25. Designed to capture mean reversion in ranging markets during both accumulation and distribution phases.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag while maintaining edge in ranging regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Williams %R (14-period), Volume Spike, Chop Index ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Volume Spike: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_1d > (1.5 * vol_ma_20)
    
    # Chop Index: measures ranging vs trending market
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # ATR(14)
    atr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Chop = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh_14 - ll_14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((hh_14 - ll_14) == 0, 50, chop)
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)
    chop_regime = chop > 61.8  # > 61.8 = ranging market (mean reversion favorable)
    
    # Align all 1d indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        wr = williams_r_aligned[i]
        vol_spike = volume_spike_aligned[i]
        is_chop = chop_regime_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50 (momentum returning)
            if wr > -50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50 (momentum returning)
            if wr < -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND volume spike AND choppy market
            if wr < -80 and vol_spike and is_chop:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 (overbought) AND volume spike AND choppy market
            elif wr > -20 and vol_spike and is_chop:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_WilliamsR_1dVolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0