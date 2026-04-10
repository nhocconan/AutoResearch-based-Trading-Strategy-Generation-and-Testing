#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 4h volume > 2.0x 20-period volume SMA AND chop > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND 4h volume > 2.0x 20-period volume SMA AND chop > 61.8 (ranging market)
# - Exit: price returns to Camarilla PIVOT level or volume drops below average
# - Position sizing: 0.30 discrete level to balance capture and fee drag
# - Target: 20-50 trades/year on 4h timeframe to stay within fee drag limits
# - Works in both bull/bear: Camarilla levels adapt to volatility, chop filter avoids false breakouts in strong trends

name = "4h_1d_camarilla_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), etc.
    # But we need the levels based on the COMPLETED 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels for each completed 1d bar
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_h3_1d = close_1d + 1.0 * range_1d
    camarilla_l3_1d = close_1d - 1.0 * range_1d
    camarilla_pivot_1d = pivot_1d
    
    # Align HTF levels to LTF (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    
    # Calculate 4h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period) for regime detection
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log10(highest_high - lowest_low)))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_14 = pd.Series(np.abs(high - low)).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = np.sum(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values) / (14 * np.log10(highest_high_14 - lowest_low_14 + 1e-10))
    chop = 100 * np.log10(chop_raw + 1e-10)  # Add small epsilon to avoid log(0)
    # Actually, let's use a more standard chop calculation
    # Reset and calculate properly
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # First bar
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = highest_high_14 - lowest_low_14
    chop = 100 * np.log10(atr_sum / (hh_ll_diff + 1e-10)) / np.log10(14)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        # Regime filter: Chop > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop[i] > 61.8
        
        # Camarilla breakout signals
        breakout_h3 = close[i] > camarilla_h3_aligned[i-1]  # Break above H3
        breakout_l3 = close[i] < camarilla_l3_aligned[i-1]  # Break below L3
        
        # Exit conditions: return to pivot or loss of volume/regime conditions
        exit_long = close[i] < camarilla_pivot_aligned[i] or not (vol_confirm and ranging_market)
        exit_short = close[i] > camarilla_pivot_aligned[i] or not (vol_confirm and ranging_market)
        
        if position == 0:  # Flat - look for entry
            if breakout_h3 and vol_confirm and ranging_market:
                position = 1
                signals[i] = 0.30
            elif breakout_l3 and vol_confirm and ranging_market:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
    
    return signals