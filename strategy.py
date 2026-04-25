#!/usr/bin/env python3
"""
1h Camarilla H3/L3 Breakout with 4h EMA34 Trend Filter and Volume Spike Confirmation
Hypothesis: Camarilla H3/L3 levels act as strong support/resistance on 4h. 
Breakouts above H3 or below L3 with volume confirmation (>2.0x 20-bar vol MA) and 
4h EMA34 trend alignment capture strong moves. Designed for 1h timeframe with 
session filter (08-20 UTC) to reduce noise and maintain 15-37 trades/year.
Uses 4h/1d for signal direction, 1h only for entry timing.
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
    
    # Get 4h data for EMA34 trend filter and Camarilla calculation (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:  # Need 34 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema_34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for Camarilla levels (previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Shift to get previous day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 1h timeframe
    prev_high_1h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_1h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_1h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels: H3, L3, H4, L4
    # H3 = Close + (High - Low) * 1.1/4
    # L3 = Close - (High - Low) * 1.1/4
    # H4 = Close + (High - Low) * 1.1/2
    # L4 = Close - (High - Low) * 1.1/2
    rang = prev_high_1h - prev_low_1h
    camarilla_h3 = prev_close_1h + rang * 1.1 / 4
    camarilla_l3 = prev_close_1h - rang * 1.1 / 4
    camarilla_h4 = prev_close_1h + rang * 1.1 / 2
    camarilla_l4 = prev_close_1h - rang * 1.1 / 2
    
    # Calculate 20-period volume MA for volume spike confirmation (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, Camarilla, and volume MA
    start_idx = max(35, 20)  # 35 for EMA34 (34 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_4h_aligned[i]
        h3 = camarilla_h3[i]
        l3 = camarilla_l3[i]
        h4 = camarilla_h4[i]
        l4 = camarilla_l4[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price above/below 4h EMA34
        price_above_ema = curr_close > ema_34_val
        price_below_ema = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Determine market regime: trending (price outside H3/L3) or ranging (price between H3/L3)
            in_range = (curr_close >= l3) and (curr_close <= h3)
            
            if in_range:
                # Ranging market: fade extremes
                long_signal = (curr_close <= l3 * 1.002) and volume_confirm  # near L3
                short_signal = (curr_close >= h3 * 0.998) and volume_confirm  # near H3
            else:
                # Trending market: breakout in direction of trend
                if price_above_ema:
                    # Uptrend: look for long breakouts above H3/H4
                    long_signal = (curr_close > h3) and volume_confirm
                else:
                    # Downtrend: look for short breakdowns below L3/L4
                    short_signal = (curr_close < l3) and volume_confirm
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.20
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.20
                position = -1
            # Clear signal flags for next iteration
            if 'long_signal' in locals():
                del long_signal
            if 'short_signal' in locals():
                del short_signal
        elif position == 1:
            # Exit long: price breaks below L3 or reverses below EMA
            if curr_close < l3 or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above H3 or reverses above EMA
            if curr_close > h3 or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0