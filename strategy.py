#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and chop regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume confirmation and chop regime detection.
- Camarilla levels: H3/L3 from prior 1d OHLC (strong intraday support/resistance).
- Regime: Chopiness index (14) > 61.8 = choppy (fade H3/L3), < 38.2 = trending (break H3/L3).
- Volume: Current 12h volume > 2.0 * 20-period average 12h volume for confirmation.
- Entry: Long when price > H3 AND trending regime AND volume confirmation.
         Short when price < L3 AND trending regime AND volume confirmation.
- Exit: Opposite Camarilla level (price < H3 for long exit, price > L3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets via breakouts and bear markets via faded H3/L3 in chop (if added later).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Chopiness Index (14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need sufficient data for CHOP
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr = np.concatenate([[np.nan], tr1])  # First TR is undefined
    
    # ATR(14) - sum of TR over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Chopiness Index: 100 * log10(ATR(14) / (max(high)-min(low) over 14)) / log10(14)
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_14 - min_low_14
    # Avoid division by zero
    chop_denom = np.where(chop_denom == 0, np.nan, chop_denom)
    chop = 100 * np.log10(atr14 / chop_denom) / np.log10(14)
    
    # Align Chopiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate prior 1d OHLC for Camarilla levels (shifted by 1 to avoid look-ahead)
    # We need prior day's OHLC, so we shift the 1d data by 1
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Shift 1d OHLC by 1 to get prior completed day
    prior_high_1d = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prior_low_1d = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    prior_close_1d = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    
    # Camarilla levels: H3 = close + (high-low)*1.1/4, L3 = close - (high-low)*1.1/4
    prior_range = prior_high_1d - prior_low_1d
    camarilla_h3 = prior_close_1d + prior_range * 1.1 / 4
    camarilla_l3 = prior_close_1d - prior_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: Chopiness < 38.2 = trending (favor breakouts), > 61.8 = choppy (favor mean reversion)
        # For breakout strategy, we only want trending markets
        trending_regime = chop_aligned[i] < 38.2
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Exit conditions: opposite Camarilla level
        if position != 0:
            # Exit long: price < H3
            if position == 1:
                if curr_close < camarilla_h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > L3
            elif position == -1:
                if curr_close > camarilla_l3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with regime and volume filters
        if position == 0:
            # Long: price > H3 AND trending regime AND volume confirmation
            long_condition = (curr_close > camarilla_h3_aligned[i] and 
                            trending_regime and
                            volume_confirm)
            
            # Short: price < L3 AND trending regime AND volume confirmation
            short_condition = (curr_close < camarilla_l3_aligned[i] and 
                             trending_regime and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dVolSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0