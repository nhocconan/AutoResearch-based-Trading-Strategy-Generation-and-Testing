#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels act as strong support/resistance. Breakouts above H3 or below L3 
with volume confirmation, aligned 1d EMA34 trend, and choppiness regime filter capture strong momentum 
while avoiding false breakouts in ranging markets. Designed for BTC/ETH with 75-200 total trades over 4 years.
Uses chop filter to avoid whipsaws in bear markets (2022, 2025+) and volume spike to confirm institutional interest.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and EMA34 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need 34+1 for EMA34 and previous day
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Camarilla levels: H3/L3 for entry, H4/L4 for stop
    camarilla_h3_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l3_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_h4_1d = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l4_1d = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate choppiness index (CHOP) regime filter on 4h
    # CHOP > 61.8 = ranging/choppy market (mean revert), CHOP < 38.2 = trending
    atr_period = 14
    chop_period = 14
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate highest high and lowest low over chop_period
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    for i in range(chop_period, n):
        hh[i] = np.max(high[i-chop_period+1:i+1])
        ll[i] = np.min(low[i-chop_period+1:i+1])
    
    # Chop = 100 * log10(sum(atr over period) / log10(hh - ll)) / log10(period)
    chop = np.full(n, np.nan)
    for i in range(chop_period, n):
        if np.isnan(atr[i]) or np.isnan(hh[i]) or np.isnan(ll[i]) or hh[i] <= ll[i]:
            continue
        sum_atr = np.sum(atr[i-chop_period+1:i+1])
        chop[i] = 100 * np.log10(sum_atr) / np.log10(chop_period) / np.log10(hh[i] - ll[i])
    
    # Chop regime: trending when CHOP < 40 (avoid ranging markets)
    chop_filter = chop < 40.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA, Camarilla, and chop
    start_idx = max(34, 20, chop_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        camarilla_h4 = camarilla_h4_aligned[i]
        camarilla_l4 = camarilla_l4_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        is_chop_filter = chop_filter[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Entry conditions: breakout + volume + trend + chop filter
        if position == 0:
            # Long: price breaks above H3 with volume confirmation in uptrend and trending regime
            long_breakout = (curr_close > camarilla_h3) and volume_confirm and uptrend and is_chop_filter
            # Short: price breaks below L3 with volume confirmation in downtrend and trending regime
            short_breakout = (curr_close < camarilla_l3) and volume_confirm and downtrend and is_chop_filter
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below L3 OR EMA34 trend turns down OR chop becomes too high (ranging)
            if curr_close < camarilla_l3 or curr_close < ema_34_val or not is_chop_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above H3 OR EMA34 trend turns up OR chop becomes too high (ranging)
            if curr_close > camarilla_h3 or curr_close > ema_34_val or not is_chop_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0