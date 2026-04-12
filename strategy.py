#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
    # Works in bull/bear by trading institutional levels with volume confirmation
    # Chop filter avoids whipsaws in ranging markets. Target: 20-50 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume context (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Camarilla pivot levels (using previous day's OHLC)
    camarilla_h4 = np.full(len(df_1d), np.nan)  # resistance
    camarilla_l4 = np.full(len(df_1d), np.nan)  # support
    camarilla_h3 = np.full(len(df_1d), np.nan)  # resistance
    camarilla_l3 = np.full(len(df_1d), np.nan)  # support
    
    for i in range(1, len(df_1d)):
        # Calculate pivot from previous day
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        
        # Camarilla formulas
        camarilla_h4[i] = close_prev + 1.5 * (high_prev - low_prev)
        camarilla_l4[i] = close_prev - 1.5 * (high_prev - low_prev)
        camarilla_h3[i] = close_prev + 1.25 * (high_prev - low_prev)
        camarilla_l3[i] = close_prev - 1.25 * (high_prev - low_prev)
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d volume spike filter: current volume > 2.0 * 20-period average
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_filter = volume > (2.0 * vol_ma_20_1d_aligned)
    
    # 1d Choppiness Index regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    # We'll use CHOP < 50 as our filter to avoid strong trends where breakouts fail
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        tr = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        if i == 1:
            atr_1d[i] = tr
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14  # 14-period ATR
    
    # Calculate highest high and lowest low over 14 periods
    hh_1d = np.full(len(df_1d), np.nan)
    ll_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        hh_1d[i] = np.max(high_1d[i-14:i+1])
        ll_1d[i] = np.min(low_1d[i-14:i+1])
    
    # Chop = 100 * log10(sum(atr14) / log10(hh14 - ll14)) / log10(14)
    sum_atr_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        sum_atr_14[i] = np.sum(atr_1d[i-14:i+1])
    
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if hh_1d[i] > ll_1d[i] and sum_atr_14[i] > 0:
            chop[i] = 100 * np.log10(sum_atr_14[i]) / np.log10(hh_1d[i] - ll_1d[i])
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_filter = chop_aligned < 50.0  # Avoid strong trending markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > h4_aligned[i]  # Break above H4 resistance
        breakout_short = close[i] < l4_aligned[i]  # Break below L4 support
        
        # Entry conditions: breakout + volume filter + chop filter
        long_entry = breakout_long and volume_filter[i] and chop_filter[i]
        short_entry = breakout_short and volume_filter[i] and chop_filter[i]
        
        # Exit conditions: opposite breakout or loss of filters
        long_exit = (close[i] < l3_aligned[i]) or (not volume_filter[i]) or (not chop_filter[i])
        short_exit = (close[i] > h3_aligned[i]) or (not volume_filter[i]) or (not chop_filter[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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

name = "4h_1d_camarilla_breakout_vol_chop_v1"
timeframe = "4h"
leverage = 1.0