#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + choppiness regime filter
# - Primary signal: 4h close breaks above Camarilla H3 (long) or below L3 (short) from prior 1d session
# - Volume confirmation: 1d volume > 1.5x 20-period average volume (ensures participation)
# - Regime filter: 1d Choppiness Index > 61.8 (range-bound market) for mean reversion at pivots
# - Position size: 0.25 (discrete level) to balance return and fee drag
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Camarilla pivots act as support/resistance in all regimes; chop filter avoids false breakouts in strong trends

name = "4h_1d_camarilla_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: H3, L3, H4, L4
    rang = prior_high - prior_low
    H3 = prior_close + rang * 1.1 / 4
    L3 = prior_close - rang * 1.1 / 4
    H4 = prior_close + rang * 1.1 / 2
    L4 = prior_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (prior day's levels available at 00:00 UTC daily)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1d volume regime: volume > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 1d Choppiness Index (CHOP) - range bound > 61.8, trending < 38.2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max/min close over 14 periods
    max_close_14 = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_close_14 = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (max_close - min_close)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = max_close_14 - min_close_14
    chop = np.where(denominator > 0,
                    100 * np.log10(sum_tr_14 / denominator) / np.log10(14),
                    50)  # neutral when no range
    
    # Chop > 61.8 = range-bound (good for mean reversion at pivots)
    chop_regime = chop > 61.8
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
            
        close_price = prices['close'].iloc[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below H3 (profit target) or above H4 (stop)
            if close_price < H3_aligned[i] or close_price > H4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above L3 (profit target) or below L4 (stop)
            if close_price > L3_aligned[i] or close_price < L4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume spike and chop regime
            # Long: close breaks above H3 AND volume spike AND chop regime (range)
            if (close_price > H3_aligned[i] and 
                volume_spike_aligned[i] and 
                chop_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: close breaks below L3 AND volume spike AND chop regime (range)
            elif (close_price < L3_aligned[i] and 
                  volume_spike_aligned[i] and 
                  chop_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals