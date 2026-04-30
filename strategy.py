#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1d volume spike filter and chop regime filter
# Camarilla R4/S4 are stronger breakout levels than R3/S3, reducing false signals
# 1d volume spike (>2.0x average) confirms institutional participation
# Choppiness index regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) avoids whipsaws
# Works in bull/bear: breakouts with volume occur in all regimes, chop filter avoids sideways whipsaws
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_Camarilla_R4S4_Breakout_1dVolumeSpike_ChopFilter_v1"
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
    
    # Calculate 12h Camarilla levels (R4, S4) from previous bar
    # R4 = Close + 1.1*(High-Low)
    # S4 = Close - 1.1*(High-Low)
    hl_range = high - low
    camarilla_r4 = close + 1.1 * hl_range
    camarilla_s4 = close - 1.1 * hl_range
    
    # Need previous bar's levels to avoid look-ahead
    camarilla_r4_prev = np.roll(camarilla_r4, 1)
    camarilla_s4_prev = np.roll(camarilla_s4, 1)
    camarilla_r4_prev[0] = np.nan
    camarilla_s4_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > camarilla_r4_prev
    breakout_down = close < camarilla_s4_prev
    
    # Calculate 1d volume spike filter (>2.0x 20-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_ma_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Calculate 12h Choppiness Index for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    # CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    atr_period = 14
    chop_period = 14
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Sum of ATR over chop_period
    sum_atr = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Max high and min low over chop_period
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(chop_period)
    chop = np.where((max_high - min_low) > 0, chop, 50.0)  # avoid division by zero
    
    # Regime filter: prefer trending markets (CHOP < 38.2) or strong breakouts in ranging markets
    # We'll allow trades in all regimes but require volume spike for confirmation
    chop_filter = (chop < 61.8)  # avoid extremely choppy markets (CHOP > 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, chop_period)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r4_prev[i]) or 
            np.isnan(camarilla_s4_prev[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_vol_spike = vol_spike_1d_aligned[i]
        curr_chop_filter = chop_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume spike and regime filter
            if curr_vol_spike and curr_chop_filter:
                # Bullish breakout: price above Camarilla R4
                if curr_breakout_up:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below Camarilla S4
                elif curr_breakout_down:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla S4 (reversal) or above Camarilla R4 (take profit)
            if curr_close < camarilla_s4_prev[i] or curr_close > camarilla_r4_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla R4 (reversal) or below Camarilla S4 (take profit)
            if curr_close > camarilla_r4_prev[i] or curr_close < camarilla_s4_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals