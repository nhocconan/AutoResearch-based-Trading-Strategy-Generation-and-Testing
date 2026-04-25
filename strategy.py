#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance in intraday trading.
Breakout above H3 with volume confirmation and 1d EMA34 uptrend signals bullish momentum.
Breakdown below L3 with volume confirmation and 1d EMA34 downtrend signals bearish momentum.
Uses 4h primary timeframe for optimal trade frequency (20-50/year) to minimize fee drift.
Designed for BTC/ETH with discrete position sizing (0.25) to control drawdown in bear markets.
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
    
    # Get 1d data for Camarilla pivots and EMA34 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need 34 for EMA34
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # H3 = C + (H-L) * 1.1/4
    # L3 = C - (H-L) * 1.1/4
    camarilla_h3 = close_1d + range_1d * 1.1 / 4.0
    camarilla_l3 = close_1d - range_1d * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe (1d -> 4h)
    # Note: Camarilla levels are based on previous day, so we shift by 1 to avoid look-ahead
    camarilla_h3_shifted = np.roll(camarilla_h3, 1)
    camarilla_l3_shifted = np.roll(camarilla_l3, 1)
    camarilla_h3_shifted[0] = np.nan  # First value has no previous day
    camarilla_l3_shifted[0] = np.nan
    
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3_shifted)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3_shifted)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = max(35, 20)  # 35 for EMA34 (34 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        h3_val = h3_4h[i]
        l3_val = l3_4h[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout above H3 (long) or breakdown below L3 (short)
            if curr_close > h3_val and volume_confirm and curr_close > ema_34_val:
                # Bullish breakout: price above H3 with volume and above EMA34 (uptrend)
                signals[i] = 0.25
                position = 1
            elif curr_close < l3_val and volume_confirm and curr_close < ema_34_val:
                # Bearish breakdown: price below L3 with volume and below EMA34 (downtrend)
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below L3 OR EMA34 turns down
            if curr_close < l3_val or ema_34_val < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above H3 OR EMA34 turns up
            if curr_close > h3_val or ema_34_val > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0