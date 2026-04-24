#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Price breaks above/below 4h Camarilla H3/L3 levels with volume > 2.0 * 20-period volume MA and 1d EMA50 alignment.
- Exit: ATR-based stoploss (1.5 * ATR(14)) or Camarilla level reversal (touch opposite L3/H3 level).
- Signal size: 0.25 discrete to balance capture and fee control.
Designed to work in both bull and bear markets by following higher timeframe trend while using lower timeframe breakouts for entry timing.
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
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (H3, L3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla calculations using previous 4h bar's OHLC
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = high_4h[0]  # first bar uses current values
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    rang = prev_high - prev_low
    camarilla_h3 = prev_close + rang * 1.1 / 4
    camarilla_l3 = prev_close - rang * 1.1 / 4
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Calculate 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(5, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_4h_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
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
            vol_confirmed = curr_volume > 2.0 * vol_ma_4h_aligned[i]
            
            # Determine 1d EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
            trend_bullish = close[i] > ema_50_aligned[i]
            trend_bearish = close[i] < ema_50_aligned[i]
            
            # Long: price breaks above Camarilla H3 AND 1d trend bullish AND volume confirmed
            if curr_high > camarilla_h3_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla L3 AND 1d trend bearish AND volume confirmed
            elif curr_low < camarilla_l3_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Camarilla L3 (reversal signal)
            stop_loss = entry_price - 1.5 * atr_4h_aligned[i]
            if curr_low < stop_loss or curr_low < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Camarilla H3 (reversal signal)
            stop_loss = entry_price + 1.5 * atr_4h_aligned[i]
            if curr_high > stop_loss or curr_high > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0