#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R extreme + 1d volume spike + chop regime filter
    # Williams %R < -80 = oversold, > -20 = overbought. Trade reversals from extremes
    # only when volume confirms and market is choppy (avoid trending whipsaws).
    # Target: 12-37 trades/year per symbol. Works in bull/bear via mean reversion.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Williams %R(14) on 1d
    highest_high_14 = np.full(len(df_1d), np.nan)
    lowest_low_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        highest_high_14[i] = np.max(high_1d[i-13:i+1])
        lowest_low_14[i] = np.min(low_1d[i-13:i+1])
    williams_r = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        denom = highest_high_14[i] - lowest_low_14[i]
        if denom != 0:
            williams_r[i] = (highest_high_14[i] - close_1d[i]) / denom * -100
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume_1d_aligned > (2.0 * vol_ma_20_1d_aligned) if 'volume_1d_aligned' in locals() else np.full(n, False)
    # Need to align volume_1d first
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    volume_spike = volume_1d_aligned > (2.0 * vol_ma_20_1d_aligned)
    
    # 1d Choppiness Index(14) for regime filter
    chop = np.full(len(df_1d), np.nan)
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = tr if i < 14 else np.mean(atr_1d[i-13:i+1]) if i >= 14 else atr_1d[i-1]
    # Initialize first 13 ATR values
    if len(atr_1d) >= 14:
        atr_1d[0] = max(high_1d[0] - low_1d[0], abs(high_1d[0] - close_1d[0]), abs(low_1d[0] - close_1d[0]))  # placeholder
        for i in range(1, 14):
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
            atr_1d[i] = np.mean(atr_1d[max(0, i-13):i+1])
    
    for i in range(13, len(df_1d)):
        atr_sum = np.sum(atr_1d[i-13:i+1])
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if atr_sum > 0 and highest_high > lowest_low:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
        else:
            chop[i] = 50.0
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_filter = chop_aligned > 61.8  # choppy regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (williams_r_aligned[i] < -80) and volume_spike[i] and chop_filter[i]
        short_entry = (williams_r_aligned[i] > -20) and volume_spike[i] and chop_filter[i]
        
        # Exit conditions: opposite extreme or regime change
        long_exit = (williams_r_aligned[i] > -20) or (not chop_filter[i])
        short_exit = (williams_r_aligned[i] < -80) or (not chop_filter[i])
        
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

name = "12h_1d_williams_r_volume_chop_v1"
timeframe = "12h"
leverage = 1.0