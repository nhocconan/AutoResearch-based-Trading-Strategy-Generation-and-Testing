#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Trade 4h Camarilla H3/L3 breakouts with 1d EMA34 trend filter and 1d volume spike (>2.0x 20-bar MA), plus choppiness regime filter (CHOP > 50 = range, mean revert at H3/L3). Uses 4h for lower trade frequency, Camarilla H3/L3 for stronger breakouts than R1/S1, volume confirmation for institutional interest, and chop filter to avoid whipsaws in strong trends. Discrete sizing 0.25. Target 20-50 trades/year on 4h timeframe. Works in bull/bear via trend filter + volume confirmation + regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d for HTF trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate choppiness index on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (max(high) - min(low)))) / log10(n)
    # Simplified: CHOP > 50 = ranging, CHOP < 50 = trending
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = 14 * (max_high_14 - min_low_14)
    chop_ratio = np.where(chop_denom != 0, sum_atr_14 / chop_denom, 1.0)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_filter = chop > 50.0  # True = ranging (mean revert), False = trending (trend follow)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # Calculate Camarilla levels from previous 4h bar
    camarilla_range = (high - low) * 1.1 / 12.0
    camarilla_H3 = close + camarilla_range * 1.125  # H3 = C + (H-L)*1.1/12*1.125
    camarilla_L3 = close - camarilla_range * 1.125  # L3 = C - (H-L)*1.1/12*1.125
    
    # Shift by 1 to use only completed 4h bar for Camarilla calculation (no look-ahead)
    camarilla_H3 = np.roll(camarilla_H3, 1)
    camarilla_L3 = np.roll(camarilla_L3, 1)
    camarilla_H3[0] = np.nan
    camarilla_L3[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20), ATR (14), and Camarilla (1)
    start_idx = max(34, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop_filter_aligned[i]) or 
            np.isnan(camarilla_H3[i]) or 
            np.isnan(camarilla_L3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # In ranging market (CHOP > 50): mean revert at H3/L3
            # In trending market (CHOP <= 50): breakout of H3/L3 with trend filter
            if chop_filter_aligned[i]:  # Ranging market - mean revert
                # Long: price breaks below L3 + above 1d EMA34 + volume spike (buy the dip in uptrend)
                long_setup = (close[i] < camarilla_L3[i]) and \
                             (close[i] > ema_34_1d_aligned[i]) and \
                             volume_spike_1d_aligned[i]
                # Short: price breaks above H3 + below 1d EMA34 + volume spike (sell the rally in downtrend)
                short_setup = (close[i] > camarilla_H3[i]) and \
                              (close[i] < ema_34_1d_aligned[i]) and \
                              volume_spike_1d_aligned[i]
            else:  # Trending market - breakout
                # Long: price breaks above H3 + above 1d EMA34 + volume spike
                long_setup = (close[i] > camarilla_H3[i]) and \
                             (close[i] > ema_34_1d_aligned[i]) and \
                             volume_spike_1d_aligned[i]
                # Short: price breaks below L3 + below 1d EMA34 + volume spike
                short_setup = (close[i] < camarilla_L3[i]) and \
                              (close[i] < ema_34_1d_aligned[i]) and \
                              volume_spike_1d_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if chop_filter_aligned[i]:  # Ranging market - exit at opposite level
                if close[i] > camarilla_H3[i]:
                    signals[i] = 0.0
                    position = 0
            else:  # Trending market - exit on trend reversal or volume dry-up
                if (close[i] < ema_34_1d_aligned[i]) or \
                   (not volume_spike_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if chop_filter_aligned[i]:  # Ranging market - exit at opposite level
                if close[i] < camarilla_L3[i]:
                    signals[i] = 0.0
                    position = 0
            else:  # Trending market - exit on trend reversal or volume dry-up
                if (close[i] > ema_34_1d_aligned[i]) or \
                   (not volume_spike_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0