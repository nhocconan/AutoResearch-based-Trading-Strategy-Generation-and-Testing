#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Price breaks above/below 4h Camarilla H3/L3 levels with volume > 2.0 * 4h volume MA(20) and 12h EMA34 alignment.
- Exit: Price touches the opposite Camarilla level (L3 for long, H3 for short) or crosses the 4h close below/above the 12h EMA34 (trend failure).
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Camarilla levels provide institutional reference points, EMA34 filters trend, volume spike confirms institutional participation.
- Works in bull markets by following trend with breakouts, works in bear markets by fading false breakouts at extremes and capturing mean reversion.
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
    
    # Get 4h data for Camarilla and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # For intraday, we use the previous 4h bar's high/low/close
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Camarilla calculations
    rang = prev_high - prev_low
    h3 = prev_close + rang * 1.1 / 4
    l3 = prev_close - rang * 1.1 / 4
    h4 = prev_close + rang * 1.1 / 2
    l4 = prev_close - rang * 1.1 / 2
    
    # Align Camarilla levels from 4h to 4h timeframe (direct use with alignment for safety)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4)
    
    # Calculate 12h EMA34 for trend
    ema_34 = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(35, 21)  # EMA34 needs 34, Camarilla needs 1 previous bar
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
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
            vol_spike = curr_volume > 2.0 * vol_ma_4h_aligned[i]
            
            # Determine 12h EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
            trend_bullish = close[i] > ema_34_aligned[i]
            trend_bearish = close[i] < ema_34_aligned[i]
            
            # Long: price breaks above H3 AND 12h trend bullish AND volume spike
            if curr_high > h3_aligned[i] and trend_bullish and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below L3 AND 12h trend bearish AND volume spike
            elif curr_low < l3_aligned[i] and trend_bearish and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on touch of L3 (mean reversion) or close below EMA34 (trend failure)
            if curr_low <= l3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on touch of H3 (mean reversion) or close above EMA34 (trend failure)
            if curr_high >= h3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0