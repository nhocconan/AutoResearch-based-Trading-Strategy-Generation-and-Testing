#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and choppiness regime filter
# - Long when price breaks above H3 (Camarilla resistance) AND volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Short when price breaks below L3 (Camarilla support) AND volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Exit when price returns to Pivot Point (PP) level
# - Uses discrete position sizing (0.30) to balance return and drawdown
# - Camarilla levels from 12h timeframe provide institutional support/resistance
# - Volume confirmation avoids low-liquidity false breakouts
# - Choppiness filter ensures we only trade in trending markets (avoid choppy whipsaws)
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in both bull and bear markets: breakouts capture trends, chop filter avoids ranging losses

name = "4h_12h_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_12h + low_12h + close_12h) / 3.0
    
    # Calculate Camarilla levels
    range_12h = high_12h - low_12h
    h3 = pp + (range_12h * 1.1 / 4.0)  # Resistance level
    l3 = pp - (range_12h * 1.1 / 4.0)  # Support level
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * volume_20_avg_12h)
    
    # Pre-compute 4h choppiness index (CHOP) for regime filter
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    max_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) == 0, 50, chop)  # Avoid division by zero
    chop = np.where(np.isnan(chop), 50, chop)  # Fill NaN with neutral value
    
    # Trending regime: CHOP < 61.8
    trending_regime = chop < 61.8
    
    # Align HTF indicators to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Entry conditions
    long_entry = (prices['close'].values > h3_aligned) & vol_spike_12h_aligned & trending_regime
    short_entry = (prices['close'].values < l3_aligned) & vol_spike_12h_aligned & trending_regime
    
    # Exit condition: price returns to pivot point (PP)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    long_exit = prices['close'].values < pp_aligned
    short_exit = prices['close'].values > pp_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_spike_12h_aligned[i]) or np.isnan(trending_regime[i]) or
            np.isnan(pp_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            if long_entry[i]:
                position = 1
                signals[i] = 0.30
            elif short_entry[i]:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point
            if position == 1:  # Long position
                if long_exit[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # Short position
                if short_exit[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals