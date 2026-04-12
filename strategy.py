#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
    # Works in bull/bear by trading institutional levels with volume confirmation
    # Chop filter avoids ranging markets. Target: 12-37 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    camarilla_high = np.full(len(df_1d), np.nan)
    camarilla_low = np.full(len(df_1d), np.nan)
    camarilla_close = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        pivot = (ph + pl + 2*pc) / 4
        camarilla_high[i] = pc + 1.1 * (ph - pl) / 2  # H4 resistance
        camarilla_low[i] = pc - 1.1 * (ph - pl) / 2   # L4 support
        camarilla_close[i] = pc
    
    # Align Camarilla levels to 12h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # 1d volume spike filter: current volume > 1.5 * 20-period average
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume > 1.5 * vol_ma_20_1d_aligned
    
    # 1d Choppiness Index regime filter: CHOP < 38.2 = trending (good for breakouts)
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        tr = max(high_1d[i] - low_1d[i], 
                abs(high_1d[i] - close_1d[i-1]), 
                abs(low_1d[i] - close_1d[i-1]))
        if i == 1:
            atr_1d[i] = tr
        else:
            atr_1d[i] = 0.9 * atr_1d[i-1] + 0.1 * tr
    
    # Calculate highest high and lowest low over 14 periods
    hh_1d = np.full(len(df_1d), np.nan)
    ll_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        hh_1d[i] = np.max(high_1d[i-13:i+1])
        ll_1d[i] = np.min(low_1d[i-13:i+1])
    
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if hh_1d[i] != ll_1d[i]:
            sum_tr = 0
            for j in range(i-13, i+1):
                tr = max(high_1d[j] - low_1d[j], 
                        abs(high_1d[j] - close_1d[j-1]), 
                        abs(low_1d[j] - close_1d[j-1]))
                sum_tr += tr
            chop_1d[i] = 100 * np.log10(sum_tr / (atr_1d[i] * 14)) / np.log10(14)
        else:
            chop_1d[i] = 50  # neutral when range is zero
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    chop_filter = chop_1d_aligned < 38.2  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_high_aligned[i]
        breakout_short = close[i] < camarilla_low_aligned[i]
        
        # Entry conditions: breakout + volume spike + trending regime
        long_entry = breakout_long and volume_spike[i] and chop_filter[i]
        short_entry = breakout_short and volume_spike[i] and chop_filter[i]
        
        # Exit conditions: opposite breakout or regime change
        long_exit = (close[i] < camarilla_low_aligned[i]) or (not chop_filter[i])
        short_exit = (close[i] > camarilla_high_aligned[i]) or (not chop_filter[i])
        
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

name = "12h_1d_camarilla_breakout_vol_chop_v1"
timeframe = "12h"
leverage = 1.0