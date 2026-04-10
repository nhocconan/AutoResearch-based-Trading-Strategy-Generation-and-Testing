#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and chop regime filter
# - Primary: 6h price action at Camarilla levels derived from 1d OHLC
# - HTF: 12h volume spike (> 2.0x 24-period MA) for breakout conviction
# - Regime: 6h Choppiness Index (CHOP < 61.8) to avoid false breakouts in ranging markets
# - Long: Price breaks above R4 (or S4 for short) with volume confirmation + trending regime
# - Short: Price breaks below S4 (or R4 for short) with volume confirmation + trending regime
# - Exit: Price returns to Camarilla H3/L3 levels or opposite Camarilla breakout
# - Position sizing: 0.25 discrete level to manage drawdown
# - Works in bull/bear: Camarilla adapts to volatility, volume filters false moves, chop regime avoids whipsaws
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_12h_camarilla_volume_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 2 or len(df_12h) < 30:  # Need enough data
        return np.zeros(n)
    
    # Pre-compute 6h data
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # Pre-compute 1d data for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pre-compute 12h data for volume confirmation
    volume_12h = df_12h['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    camarilla_h5 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    camarilla_l5 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            rng = high_1d[i-1] - low_1d[i-1]
            camarilla_h5[i] = close_1d[i-1] + (rng * 1.500)
            camarilla_h4[i] = close_1d[i-1] + (rng * 1.250)
            camarilla_h3[i] = close_1d[i-1] + (rng * 1.166)
            camarilla_l3[i] = close_1d[i-1] - (rng * 1.166)
            camarilla_l4[i] = close_1d[i-1] - (rng * 1.250)
            camarilla_l5[i] = close_1d[i-1] - (rng * 1.500)
    
    # Calculate 12h volume moving average (24-period) for volume confirmation
    volume_ma_24_12h = np.full(len(volume_12h), np.nan)
    for i in range(23, len(volume_12h)):
        if not np.isnan(volume_12h[i-23:i+1]).any():
            volume_ma_24_12h[i] = np.mean(volume_12h[i-23:i+1])
    
    # Calculate 6h Choppiness Index (CHOP) - 14 period
    chop = np.full(len(close_6h), np.nan)
    atr_14 = np.full(len(close_6h), np.nan)
    
    # First calculate True Range and ATR(14)
    tr = np.full(len(close_6h), np.nan)
    for i in range(1, len(close_6h)):
        if not (np.isnan(high_6h[i]) or np.isnan(low_6h[i]) or np.isnan(close_6h[i-1])):
            tr[i] = max(
                high_6h[i] - low_6h[i],
                abs(high_6h[i] - close_6h[i-1]),
                abs(low_6h[i] - close_6h[i-1])
            )
    
    # Calculate ATR(14) using Wilder's smoothing
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            if i == 14:
                atr_14[i] = np.mean(tr[1:15])  # First ATR is simple average
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14)/log10(n)) / log10(n)
    for i in range(27, len(close_6h)):  # Need 14 ATR + 14 period for CHOP
        if not np.isnan(atr_14[i-13:i+1]).any():
            sum_atr = np.sum(atr_14[i-13:i+1])
            if sum_atr > 0 and close_6h[i] > 0:
                max_high = np.max(high_6h[i-13:i+1])
                min_low = np.min(low_6h[i-13:i+1])
                if max_high > min_low:
                    chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(max_high - min_low)
    
    # Align all HTF indicators to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_24_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_24_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):  # Start after warmup period for all indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_l5_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_24_12h_aligned[i]) or 
            np.isnan(volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 24-period MA
        volume_confirm = volume_12h_aligned[i] > 2.0 * volume_ma_24_12h_aligned[i]
        
        # Chop regime filter: only trade when market is trending (CHOP < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H4 (or H5) with volume confirmation + trending regime
            if close_6h[i] > camarilla_h4_aligned[i] and volume_confirm and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L4 (or L5) with volume confirmation + trending regime
            elif close_6h[i] < camarilla_l4_aligned[i] and volume_confirm and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to H3/L3 levels OR opposite Camarilla breakout
            if position == 1:  # Long position
                if (close_6h[i] < camarilla_h3_aligned[i] or 
                    close_6h[i] > camarilla_h5_aligned[i]):  # Exit on H3 return or H5 breakout
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if (close_6h[i] > camarilla_l3_aligned[i] or 
                    close_6h[i] < camarilla_l5_aligned[i]):  # Exit on L3 return or L5 breakout
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals