#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime_v2
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakouts with 1d EMA34 trend filter, strict volume confirmation (3.0x), and choppiness regime filter (CHOP < 40) produce high-quality trades with low frequency. The tight regime filter avoids whipsaws in ranging markets, improving performance in both bull and bear regimes. Target: 60-100 total trades over 4 years (15-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter (EMA34) and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (R1, S1) using previous 1d's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.concatenate([[np.nan], close_1d[:-1]])  # previous 1d close
    
    camarilla_range = high_1d - low_1d
    r1 = close_1d_shifted + 1.1 * camarilla_range / 12
    s1 = close_1d_shifted - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 12h volume confirmation: volume > 3.0x 20-period average (very strict)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h choppiness index: CHOP < 40 = strongly trending (favor breakouts), CHOP > 61.8 = ranging (avoid)
    tr1 = np.maximum(high - low, np.absolute(high - np.concatenate([[np.nan], close[:-1]])))
    tr2 = np.maximum(tr1, np.absolute(low - np.concatenate([[np.nan], close[:-1]])))
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / (max_high_14 - min_low_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) == 0, 50, chop)  # avoid div by zero
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA, 14*2 for chop)
    start_idx = max(34, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (EMA34)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (very strict)
        volume_spike = volume[i] > 3.0 * vol_ma_20[i]
        
        # Choppiness regime: only take breakouts when CHOP < 40 (strongly trending)
        regime_ok = chop[i] < 40.0
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Long logic: breakout above R1 in uptrend with volume and good regime
        if uptrend and volume_spike and breakout_r1 and regime_ok:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S1 in downtrend with volume and good regime
        elif downtrend and volume_spike and breakout_s1 and regime_ok:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend OR regime becomes too choppy (CHOP >= 61.8)
        elif position == 1 and (not uptrend or chop[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend or chop[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime_v2"
timeframe = "12h"
leverage = 1.0