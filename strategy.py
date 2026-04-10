#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1d choppiness regime filter
# - Long when price breaks above Camarilla R4 (1d) AND 1d volume > 1.8x 24-bar avg AND 1d chop > 61.8 (ranging market)
# - Short when price breaks below Camarilla S4 (1d) AND 1d volume > 1.8x 24-bar avg AND 1d chop > 61.8 (ranging market)
# - Exit when price returns to Camarilla PP (pivot point) from 1d
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla levels provide precise support/resistance; volume confirms institutional participation
# - Choppiness filter ensures trades occur in ranging markets where mean reversion to pivot works best
# - Works in both bull and bear markets by fading extremes in ranging conditions

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from 1d data (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2.0)
    camarilla_s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 1d volume confirmation: > 1.8x 24-period average
    volume_1d = df_1d['volume'].values
    volume_24_avg = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    vol_spike_1d = volume_1d > (1.8 * volume_24_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 1d Choppiness Index: CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) * (highest_high - lowest_low)) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (n * log10(high_low_range))) / log10(n)
    # We'll use a practical approximation: CHOP = 100 * log10(ATR(14).sum(14) / (14 * log10(highest_high - lowest_low))) / log10(14)
    # But for efficiency, we'll use: CHOP > 61.8 when market is ranging (we'll calculate properly)
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    high_low_range = highest_high_14 - lowest_low_14
    high_low_range = np.where(high_low_range == 0, 1e-10, high_low_range)  # Prevent div by zero
    
    chop = 100 * np.log10(atr_sum / (14 * np.log10(high_low_range))) / np.log10(14)
    chop_regime = chop > 61.8  # True when ranging/choppy
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(chop_regime_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Camarilla R4 AND 1d volume spike AND choppy market
            if (prices['high'].iloc[i] > camarilla_r4_aligned[i] and 
                vol_spike_1d_aligned[i] and 
                chop_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Camarilla S4 AND 1d volume spike AND choppy market
            elif (prices['low'].iloc[i] < camarilla_s4_aligned[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Camarilla PP (mean reversion to equilibrium)
            # Exit when price returns to Camarilla pivot point
            exit_signal = False
            if position == 1:  # Long position
                if prices['low'].iloc[i] <= camarilla_pp_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['high'].iloc[i] >= camarilla_pp_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals