#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- HTF: 1d for Camarilla pivot levels (R1, S1) using prior day's OHLC.
- Entry: Price breaks above/below 1h Camarilla R1/S1 levels with volume > 1.8 * 20-period volume MA and 4h EMA34 alignment.
- Exit: Price touches opposite Camarilla level (R1 for shorts, S1 for longs) or 4h EMA34 reversal.
- Signal size: 0.20 discrete to minimize fee churn and control drawdown.
- Session filter: 08-20 UTC to avoid low-volume Asian session noise.
Works in both bull and bear markets by following 4h trend while using 1h Camarilla breakouts for precise entries.
Volume spike filter reduces false breakouts in choppy markets. Camarilla pivots provide structure in ranging markets.
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
    
    # Pre-compute session hours filter
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot levels (prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend
    close_4h = df_4h['close'].values
    ema_34 = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    
    # Calculate 1d Camarilla pivot levels (R1, S1) from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First value has no prior day
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla pivot calculations
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)  # R1 = pivot + (HL * 1.1/12)
    s1 = pivot - (range_hl * 1.1 / 12.0)  # S1 = pivot - (HL * 1.1/12)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1h volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1  # +1 for prior day shift
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Volume confirmation (1.8x threshold)
            vol_confirmed = curr_volume > 1.8 * vol_ma[i]
            
            # Determine 4h EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
            trend_bullish = close[i] > ema_34_aligned[i]
            trend_bearish = close[i] < ema_34_aligned[i]
            
            # Long: price breaks above Camarilla R1 AND 4h trend bullish AND volume confirmed
            if curr_high > r1_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S1 AND 4h trend bearish AND volume confirmed
            elif curr_low < s1_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on Camarilla S1 touch or 4h EMA34 bearish reversal
            if curr_low <= s1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit on Camarilla R1 touch or 4h EMA34 bullish reversal
            if curr_high >= r1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_4hEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0