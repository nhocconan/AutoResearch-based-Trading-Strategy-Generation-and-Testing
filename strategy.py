#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + 1w chop regime filter
    # Camarilla levels provide precise intraday support/resistance from prior 1d range
    # Volume spike confirms institutional participation in breakout
    # Weekly chop filter avoids trading in strong trends where pivots fail
    # Works in bull/bear by fading extremes in ranging markets
    # Target: 20-50 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for current 4h bar using prior 1d bar
    # H-L = prior day's range
    hl = high_1d[:-1] - low_1d[:-1]  # shift by 1 to use prior day
    hl = np.append([np.nan], hl)  # first bar has no prior
    
    # Camarilla levels
    camarilla_h4 = close_1d[:-1] + hl * 1.1/2  # resistance 4
    camarilla_l4 = close_1d[:-1] - hl * 1.1/2  # support 4
    camarilla_h3 = close_1d[:-1] + hl * 1.1/4  # resistance 3
    camarilla_l3 = close_1d[:-1] - hl * 1.1/4  # support 3
    camarilla_h2 = close_1d[:-1] + hl * 1.1/6  # resistance 2
    camarilla_l2 = close_1d[:-1] - hl * 1.1/6  # support 2
    camarilla_h1 = close_1d[:-1] + hl * 1.1/12 # resistance 1
    camarilla_l1 = close_1d[:-1] - hl * 1.1/12 # support 1
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Calculate 1d volume average (20-period) for spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 1w data for chop regime filter (Choppiness Index)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range for 1w
    tr_1w = np.zeros(len(df_1w))
    for i in range(len(df_1w)):
        if i == 0:
            tr_1w[i] = high_1w[i] - low_1w[i]
        else:
            tr_1w[i] = max(
                high_1w[i] - low_1w[i],
                abs(high_1w[i] - close_1w[i-1]),
                abs(low_1w[i] - close_1w[i-1])
            )
    
    # Calculate Choppiness Index (14-period)
    chop_14 = np.full(len(df_1w), np.nan)
    atr_sum_14 = np.full(len(df_1w), np.nan)
    
    for i in range(13, len(df_1w)):
        if i == 13:
            atr_sum_14[i] = np.sum(tr_1w[i-13:i+1])
        else:
            atr_sum_14[i] = atr_sum_14[i-1] - tr_1w[i-14] + tr_1w[i]
        
        if atr_sum_14[i] > 0:
            hh = np.max(high_1w[i-13:i+1])
            ll = np.min(low_1w[i-13:i+1])
            if hh > ll:
                chop_14[i] = 100 * np.log10(atr_sum_14[i] / np.log(14) / (hh - ll))
    
    # Align chop to 4h timeframe
    chop_14_aligned = align_htf_to_ltf(prices, df_1w, chop_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_14_aligned[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Chop regime: trade only when market is ranging (CHOP > 50)
        chop_regime = chop_14_aligned[i] > 50
        
        # Camarilla breakout logic
        # Long when price breaks above H3 with volume spike
        # Short when price breaks below L3 with volume spike
        long_breakout = (close[i] > camarilla_h3_aligned[i]) and volume_spike and chop_regime
        short_breakout = (low[i] < camarilla_l3_aligned[i]) and volume_spike and chop_regime
        
        # Exit when price returns to mean (L3/H3) or opposite level touched
        long_exit = (position == 1 and (close[i] < camarilla_l3_aligned[i] or 
                                       low[i] < camarilla_l4_aligned[i]))
        short_exit = (position == -1 and (close[i] > camarilla_h3_aligned[i] or 
                                         high[i] > camarilla_h4_aligned[i]))
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0