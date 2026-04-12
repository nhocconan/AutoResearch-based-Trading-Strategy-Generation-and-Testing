#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + chop regime filter
    # Camarilla levels from daily high/low provide institutional support/resistance
    # Volume spike confirms institutional participation
    # Chop filter avoids whipsaws in ranging markets
    # Works in bull/bear by trading breakouts in direction of higher timeframe trend
    # Target: 20-50 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each day
    # H4 = Close + 1.1*(High-Low)/2
    # L4 = Close - 1.1*(High-Low)/2
    # H3 = Close + 1.1*(High-Low)/4
    # L3 = Close - 1.1*(High-Low)/4
    # H2 = Close + 1.1*(High-Low)/6
    # L2 = Close - 1.1*(High-Low)/6
    # H1 = Close + 1.1*(High-Low)/12
    # L1 = Close - 1.1*(High-Low)/12
    
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h2 = np.full(len(df_1d), np.nan)
    camarilla_l2 = np.full(len(df_1d), np.nan)
    camarilla_h1 = np.full(len(df_1d), np.nan)
    camarilla_l1 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 1:  # Need at least one day
            continue
        daily_range = high_1d[i] - low_1d[i]
        camarilla_h4[i] = close_1d[i] + 1.1 * daily_range / 2
        camarilla_l4[i] = close_1d[i] - 1.1 * daily_range / 2
        camarilla_h3[i] = close_1d[i] + 1.1 * daily_range / 4
        camarilla_l3[i] = close_1d[i] - 1.1 * daily_range / 4
        camarilla_h2[i] = close_1d[i] + 1.1 * daily_range / 6
        camarilla_l2[i] = close_1d[i] - 1.1 * daily_range / 6
        camarilla_h1[i] = close_1d[i] + 1.1 * daily_range / 12
        camarilla_l1[i] = close_1d[i] - 1.1 * daily_range / 12
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Calculate 1d ATR14 for volatility filter and stoploss
    tr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i == 0:
            tr_1d[i] = high_1d[i] - low_1d[i]
        else:
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    atr14_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if i == 13:
            atr14_1d[i] = np.mean(tr_1d[i-13:i+1])
        else:
            atr14_1d[i] = (atr14_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 1d volume average for volume spike detection
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        if i == 19:
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
        else:
            vol_ma_20[i] = (vol_ma_20[i-1] * 19 + volume_1d[i]) / 20
    
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d chop regime filter (Choppiness Index)
    # CHOP = 100 * log10(sum(atr14) / (max(high) - min(low))) / log10(14)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if i >= 13:
            sum_atr = np.sum(tr_1d[i-13:i+1])
            max_high = np.max(high_1d[i-13:i+1])
            min_low = np.min(low_1d[i-13:i+1])
            if max_high > min_low and sum_atr > 0:
                chop_1d[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
            else:
                chop_1d[i] = 50  # Neutral value
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: trade only when volume is above average
        volume_spike = volume[i] > vol_ma_20_aligned[i] * 1.5
        
        # Chop regime filter: avoid extreme chop (CHOP > 61.8) and extreme trend (CHOP < 38.2)
        # We want moderate chop for mean reversion near pivots
        chop_filter = (chop_1d_aligned[i] >= 38.2) and (chop_1d_aligned[i] <= 61.8)
        
        # Long signals: price breaks above H3 or H4 with volume
        long_breakout = (close[i] > h3_aligned[i] or close[i] > h4_aligned[i]) and volume_spike
        
        # Short signals: price breaks below L3 or L4 with volume
        short_breakout = (close[i] < l3_aligned[i] or close[i] < l4_aligned[i]) and volume_spike
        
        # Exit conditions: price returns to middle levels or opposite breakout
        long_exit = close[i] < h1_aligned[i] or (close[i] < l3_aligned[i] and volume_spike)
        short_exit = close[i] > l1_aligned[i] or (close[i] > h3_aligned[i] and volume_spike)
        
        # Stoploss: ATR-based
        if position == 1 and close[i] < h4_aligned[i] - 2.0 * atr14_1d_aligned[i]:
            long_exit = True
        if position == -1 and close[i] > l4_aligned[i] + 2.0 * atr14_1d_aligned[i]:
            short_exit = True
        
        if long_breakout and chop_filter and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and chop_filter and position != -1:
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

name = "4h_1d_camarilla_breakout_vol_chop_v2"
timeframe = "4h"
leverage = 1.0