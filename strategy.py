#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume average and choppiness calculation.
- Camarilla levels: identifies key support/resistance from prior 1d session.
- Entry: Long when price breaks above R1 with volume confirmation and chop > 61.8 (rangy market).
         Short when price breaks below S1 with volume confirmation and chop > 61.8.
- Exit: Opposite Camarilla breakout or chop < 38.2 (trending market) to avoid false breakouts.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets as it captures mean reversion in ranging markets
  and avoids trending markets where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d choppiness index for regime filter
    if len(df_1d) < 14:  # Need sufficient data for chop
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).sum().values
    
    # Chop = 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    max_hh_14 = df_1d['high'].rolling(window=14, min_periods=14).max().values
    min_ll_14 = df_1d['low'].rolling(window=14, min_periods=14).min().values
    denominator = max_hh_14 - min_ll_14
    denominator = np.where(denominator == 0, 1e-10, denominator)  # Avoid division by zero
    chop = 100 * (np.log10(atr_14 / denominator) / np.log10(14))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels from 1d data (using prior day's OHLC)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4,
    #            R2 = close + 1.1*(high-low)*1.1/6, R1 = close + 1.1*(high-low)*1.1/12
    #            S1 = close - 1.1*(high-low)*1.1/12, S2 = close - 1.1*(high-low)*1.1/6,
    #            S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # We use prior day's data to avoid look-ahead
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Calculate R1 and S1 levels
    camarilla_r1 = prior_close + 1.1 * (prior_high - prior_low) * (1.1 / 12)
    camarilla_s1 = prior_close - 1.1 * (prior_high - prior_low) * (1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions
        if position != 0:
            # Exit if chop < 38.2 (trending market) - avoid false breakouts in trends
            if chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
                continue
            # Exit opposite Camarilla breakout
            elif position == 1:
                if curr_low <= s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            elif position == -1:
                if curr_high >= r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and chop > 61.8 (rangy market)
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= r1_aligned[i] and prev_close < r1_aligned[i]
            breakout_down = curr_low <= s1_aligned[i] and prev_close > s1_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Chop regime filter: chop > 61.8 (rangy market)
            chop_regime = chop_aligned[i] > 61.8
            
            if breakout_up and volume_confirm and chop_regime:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and chop_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0