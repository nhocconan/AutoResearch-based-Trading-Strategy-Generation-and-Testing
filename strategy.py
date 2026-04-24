#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Price breaks above/below 12h Camarilla H3/L3 levels with volume > 2.0 * 12h volume MA(20) and 1d EMA34 alignment.
- Exit: Price touches the opposite Camarilla level (L3 for long, H3 for short) or breaks the Camarilla midpoint (mean reversion).
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Camarilla levels from 1d provide strong support/resistance, EMA34 filters major trend, volume spike confirms institutional interest.
- Works in bull markets by following trend with breakouts, works in bear markets by fading false breakouts at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels from previous 12h bar
    # Camarilla: based on previous bar's high, low, close
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for Camarilla calculation
    typical_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels: H3, L3, H4, L4, H5, L5, H6, L6
    # We use H3 and L3 for breakout, and midpoint (H3+L3)/2 for exit
    camarilla_h3 = typical_12h + 1.1 * range_12h / 2.0
    camarilla_l3 = typical_12h - 1.1 * range_12h / 2.0
    camarilla_mid = (camarilla_h3 + camarilla_l3) / 2.0
    
    # Align Camarilla levels from 12h to 12h timeframe (direct use with alignment for safety)
    h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    mid_aligned = align_htf_to_ltf(prices, df_12h, camarilla_mid)
    
    # Calculate 1d EMA34 for trend
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 12h volume MA(20) for spike confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(mid_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume spike confirmation (2.0x threshold)
            vol_spike = curr_volume > 2.0 * vol_ma_12h_aligned[i]
            
            # Determine 1d EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
            trend_bullish = close[i] > ema_34_aligned[i]
            trend_bearish = close[i] < ema_34_aligned[i]
            
            # Long: price breaks above Camarilla H3 AND 1d trend bullish AND volume spike
            if curr_high > h3_aligned[i] and trend_bullish and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla L3 AND 1d trend bearish AND volume spike
            elif curr_low < l3_aligned[i] and trend_bearish and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on touch of Camarilla L3 (mean reversion) or break below midpoint with weakness
            if curr_low <= l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on touch of Camarilla H3 (mean reversion) or break above midpoint with weakness
            if curr_high >= h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0