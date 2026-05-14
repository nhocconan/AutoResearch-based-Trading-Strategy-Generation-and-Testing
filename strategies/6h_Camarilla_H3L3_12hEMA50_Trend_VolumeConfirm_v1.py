#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Price breaks above/below 6h Camarilla H3/L3 levels with volume > 2.0 * 20-period volume MA and 12h EMA50 alignment.
- Exit: ATR-based stoploss (2.0 * ATR(14)) or Camarilla level reversal (touch opposite level).
- Signal size: 0.25 discrete to balance capture and fee control.
Designed to work in both bull and bear markets by following higher timeframe trend while using lower timeframe breakouts for entry timing.
Volume spike filter reduces false breakouts in choppy markets.
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
    
    # Get 6h data for Camarilla levels and ATR
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 6h Camarilla levels (based on previous 6h bar's range)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate pivot and ranges from previous bar
    prev_close = np.roll(close_6h, 1)
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close[0] = close_6h[0]  # avoid NaN on first bar
    prev_high[0] = high_6h[0]
    prev_low[0] = low_6h[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4.0)
    l3 = pivot - (range_hl * 1.1 / 4.0)
    h4 = pivot + (range_hl * 1.1 / 2.0)
    l4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (already on 6h, but ensure proper alignment)
    h3_aligned = align_htf_to_ltf(prices, df_6h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_6h, l3)
    h4_aligned = align_htf_to_ltf(prices, df_6h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_6h, l4)
    
    # Calculate 12h EMA50 for trend
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 6h ATR(14) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_6h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_6h_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma_6h_aligned[i]
            
            # Determine 12h EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
            trend_bullish = close[i] > ema_50_aligned[i]
            trend_bearish = close[i] < ema_50_aligned[i]
            
            # Long: price breaks above Camarilla H3 level AND 12h trend bullish AND volume confirmed
            if curr_high > h3_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla L3 level AND 12h trend bearish AND volume confirmed
            elif curr_low < l3_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Camarilla L3 level (reversal signal)
            stop_loss = entry_price - 2.0 * atr_6h_aligned[i]
            if curr_low < stop_loss or curr_low < l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Camarilla H3 level (reversal signal)
            stop_loss = entry_price + 2.0 * atr_6h_aligned[i]
            if curr_high > stop_loss or curr_high > h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0