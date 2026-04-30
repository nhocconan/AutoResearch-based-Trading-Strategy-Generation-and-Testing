#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w volume spike and choppiness regime filter.
# Long when KAMA direction is up, 1w volume > 2.0x 20-bar average, and CHOP > 61.8 (range) for mean reversion longs near lows.
# Short when KAMA direction is down, 1w volume > 2.0x 20-bar average, and CHOP > 61.8 (range) for mean reversion shorts near highs.
# Uses KAMA adaptive trend filter from 1d timeframe to avoid whipsaws.
# Volume confirmation from 1w timeframe ensures institutional participation.
# Choppiness regime filter (CHOP > 61.8) identifies ranging markets where mean reversion works best.
# Discrete position sizing at ±0.25 to balance performance and fee drag.
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag.

name = "1d_KAMA_VolumeSpike_Chop_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for volume and choppiness
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend direction
    # KAMA parameters: ER period=10, fastest EMA=2, slowest EMA=30
    close_1d = close  # since we're on 1d timeframe
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if False else None  # placeholder
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            direction = np.abs(close_1d[i] - close_1d[i-10])
            volatility = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
            er[i] = direction / volatility if volatility > 0 else 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_dir = np.zeros_like(close_1d)
    kama_dir[1:] = np.where(kama[1:] > kama[:-1], 1, -1)
    kama_dir_aligned = kama_dir  # already on 1d
    
    # Calculate 1w volume confirmation: volume > 2.0x 20-period average
    vol_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1w = vol_1w > (2.0 * vol_ma_20_1w)
    volume_confirm_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm_1w.astype(float))
    
    # Calculate 1w Choppiness Index: CHOP > 61.8 = ranging (mean reversion zone)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    atr_1w = np.zeros_like(close_1w := df_1w['close'].values)
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), np.abs(low_1w[1:] - close_1w[:-1])))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    max_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_1w = 100 * np.log10(atr_1w * 14 / np.log(max_high_1w - min_low_1w)) / np.log10(100)
    chop_1w = np.where((max_high_1w - min_low_1w) > 0, chop_1w, 50)  # avoid division by zero
    chop_regime = chop_1w > 61.8  # True = ranging market
    chop_regime_aligned = align_htf_to_ltf(prices, df_1w, chop_regime.astype(float))
    
    # Price position within 1w range for mean reversion signals
    # Long when price near low of 1w range in ranging market
    # Short when price near high of 1w range in ranging market
    price_position_1w = (close_1w - min_low_1w) / (max_high_1w - min_low_1w + 1e-10)
    price_position_1w_aligned = align_htf_to_ltf(prices, df_1w, price_position_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(kama_dir_aligned[i]) or 
            np.isnan(volume_confirm_1w_aligned[i]) or 
            np.isnan(chop_regime_aligned[i]) or
            np.isnan(price_position_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_kama_dir = kama_dir_aligned[i]
        curr_volume_confirm = volume_confirm_1w_aligned[i] > 0.5
        curr_chop_regime = chop_regime_aligned[i] > 0.5
        curr_price_pos = price_position_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: KAMA up, volume spike, ranging market, price near low of 1w range
            if (curr_kama_dir == 1 and 
                curr_volume_confirm and 
                curr_chop_regime and 
                curr_price_pos < 0.2):  # near low of weekly range
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, volume spike, ranging market, price near high of 1w range
            elif (curr_kama_dir == -1 and 
                  curr_volume_confirm and 
                  curr_chop_regime and 
                  curr_price_pos > 0.8):  # near high of weekly range
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: KAMA turns down OR price moves to middle of range
            if curr_kama_dir == -1 or curr_price_pos > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: KAMA turns up OR price moves to middle of range
            if curr_kama_dir == 1 or curr_price_pos < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals