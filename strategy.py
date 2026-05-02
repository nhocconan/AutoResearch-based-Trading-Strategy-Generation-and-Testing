#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter
# Camarilla pivot levels (R1/S1) derived from daily OHLC provide intraday support/resistance
# Breakout above R1 or below S1 with volume confirmation (>2x 20-bar EMA) indicates momentum
# Choppiness regime filter: only trade when CHOP(14) > 61.8 (range market) for mean reversion at pivot levels
# Designed for 4h timeframe targeting 20-50 trades/year (75-200 total over 4 years)
# Uses discrete position sizing (0.30) to balance profit potential and fee drag
# Works in bull markets (mean reversion from R1 in range) and bear markets (mean reversion from S1 in range)

name = "4h_Camarilla_R1S1_Breakout_1dVolume_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels, volume confirmation, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for choppiness calculation
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar (based on same day's OHLC)
    # Standard Camarilla: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_r1 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12
    camarilla_s1 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (use same day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2x 20-bar EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    # Choppiness Index: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['high'])).abs()
    tr4 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1)
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    ll_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    # Choppiness formula: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    chop = 100 * np.log10(tr_sum_14 / (hh_14 - ll_14)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    chop_filter = chop_aligned > 61.8  # Only trade in range markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_confirmation[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R1 with volume confirmation and in range market
            if close[i] > camarilla_r1_aligned[i] and volume_confirmation[i] and chop_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: Breakout below S1 with volume confirmation and in range market
            elif close[i] < camarilla_s1_aligned[i] and volume_confirmation[i] and chop_filter[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S1 (reversal to mean) OR volume confirmation lost
            if close[i] < camarilla_s1_aligned[i] or not volume_confirmation[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R1 (reversal to mean) OR volume confirmation lost
            if close[i] > camarilla_r1_aligned[i] or not volume_confirmation[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals