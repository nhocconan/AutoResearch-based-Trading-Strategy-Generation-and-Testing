#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ChopFilter
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter, volume confirmation (>2x average), and Choppiness Index regime filter (CHOP > 61.8 = range -> mean revert, CHOP < 38.2 = trending -> trend follow). Uses discrete position sizing (0.25) to minimize fee churn. Designed to work in both bull and bear markets by following 12h trend direction and avoiding whipsaws in high-chop regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need warmup for EMA, volume, and chop
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels (tighter breakout = fewer trades)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (1 bar delay for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 4h data (14-period)
    # CHOP = 100 * log10(sum(ATR) / (HHV - LLV)) / log10(period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = np.zeros_like(close)
    mask = (hh - ll) > 0
    chop_raw[mask] = 100 * np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum().values[mask] / (hh[mask] - ll[mask])) / np.log10(14)
    chop_raw[~mask] = 50  # Neutral when HHV == LLV
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50 for EMA, 20 for volume, 14 for chop/ATR)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_12h_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        chop_val = chop_raw[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(chop_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2x average volume (strong breakout)
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Choppiness regime filter
        # CHOP > 61.8 = ranging market -> mean revert (fade breakouts)
        # CHOP < 38.2 = trending market -> trend follow (breakout continuation)
        # 38.2 <= CHOP <= 61.8 = transition -> no trade
        chop_regime_long = chop_val < 38.2  # Only take long breakouts in trending market
        chop_regime_short = chop_val < 38.2  # Only take short breakouts in trending market
        chop_regime_exit = chop_val > 61.8   # Exit if market becomes ranging
        
        # Long logic: price breaks above Camarilla R1 with 12h uptrend, volume confirmation, and trending regime
        long_condition = (close_val > r1_val) and (close_val > ema_val) and volume_confirmed and chop_regime_long
        # Short logic: price breaks below Camarilla S1 with 12h downtrend, volume confirmation, and trending regime
        short_condition = (close_val < s1_val) and (close_val < ema_val) and volume_confirmed and chop_regime_short
        
        # Exit logic: trend reversal OR chop regime shift to ranging
        exit_long = close_val < ema_val or chop_regime_exit
        exit_short = close_val > ema_val or chop_regime_exit
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0