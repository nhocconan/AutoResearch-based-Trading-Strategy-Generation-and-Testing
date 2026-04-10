#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average AND chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average AND chop < 61.8 (trending)
# - Exit when price returns to Camarilla PIVOT point (mean reversion to equilibrium)
# - Uses 1d volume confirmation to ensure institutional participation
# - Uses chop regime filter to avoid whipsaws in ranging markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots work well in both bull and bear markets by identifying key reversal levels
# - Volume confirmation filters out false breakouts
# - Chop regime filter ensures we only trade in trending conditions where breakouts are more reliable

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    camarilla_h3 = pivot + (range_1d * 1.1 / 4)
    camarilla_l3 = pivot - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute chop regime filter on 12h data (trending when chop < 61.8)
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    
    # True Range
    tr1 = prices['high'] - prices['low']
    tr2 = abs(prices['high'] - prices['close'].shift(1))
    tr3 = abs(prices['low'] - prices['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = high_14
    lowest_low = low_14
    
    # Avoid division by zero
    price_range = highest_high - lowest_low
    chop_raw = np.zeros_like(price_range)
    mask = (price_range > 0) & (atr_14 > 0) & (sum_atr_14 > 0)
    chop_raw[mask] = 100 * np.log10(sum_atr_14[mask] / price_range[mask]) / np.log10(14)
    chop = chop_raw
    
    # Chop regime: trending when chop < 61.8
    chop_regime = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(chop_regime[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Camarilla H3 AND volume spike AND trending regime
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                vol_spike_1d_aligned[i] and 
                chop_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Camarilla L3 AND volume spike AND trending regime
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_regime[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Camarilla PIVOT (mean reversion)
            # Exit when price returns to Camarilla PIVOT point
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= pivot_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= pivot_aligned[i]:
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