#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and chop regime filter
# - Long when price breaks above H3 Camarilla level (from prior 1d) AND volume > 1.5x 20-period average AND chop < 61.8 (trending)
# - Short when price breaks below L3 Camarilla level (from prior 1d) AND volume > 1.5x 20-period average AND chop < 61.8 (trending)
# - Exit when price returns to H4/L4 levels or chop > 61.8 (range) as risk control
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Camarilla levels provide institutional support/resistance; breakouts capture momentum
# - Volume confirmation ensures institutional participation
# - Chop filter (14-period) avoids whipsaws in ranging markets
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts work in trends, chop filter avoids false signals in ranges

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
    
    # Pre-compute 1d Camarilla levels (based on prior day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior day's range (shifted by 1 to avoid look-ahead)
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    
    # Set first value to NaN (no prior day available)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Calculate Camarilla levels for current day (based on prior day)
    range_1d = prior_high - prior_low
    H3 = prior_close + (range_1d * 1.1 / 4)
    L3 = prior_close - (range_1d * 1.1 / 4)
    H4 = prior_close + (range_1d * 1.1 / 2)
    L4 = prior_close - (range_1d * 1.1 / 2)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Pre-compute 12h chop regime filter (14-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop = np.where(range_14 > 0, 100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)
    chop_filter = chop < 61.8  # Trending regime (chop < 61.8)
    
    # Align HTF indicators to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_filter[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND volume spike AND trending regime
            if (close_12h[i] > H3_aligned[i] and 
                vol_spike_1d_aligned[i] and 
                chop_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND volume spike AND trending regime
            elif (close_12h[i] < L3_aligned[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_filter[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit when price returns to H4/L4 levels OR chop > 61.8 (range)
            exit_signal = False
            if position == 1:  # Long position
                if close_12h[i] < H4_aligned[i] or chop_filter[i] == False:
                    exit_signal = True
            else:  # Short position
                if close_12h[i] > L4_aligned[i] or chop_filter[i] == False:
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