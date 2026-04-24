#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator strategy with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Price breaks above/below Williams Alligator lines (Jaw/Teeth/Lips) with volume > 2.0 * 20-period volume MA and 1d EMA50 alignment.
- Exit: ATR-based stoploss (2.0 * ATR(14)) or Williams Alligator reversal (price crosses middle line - Teeth).
- Signal size: 0.25 discrete to balance capture and fee control.
Designed to work in both bull and bear markets by following higher timeframe trend while using Williams Alligator for trend identification and lower timeframe breakouts for entry timing.
Williams Alligator is effective in ranging markets (common in bear markets) and can catch trends when they emerge.
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
    
    # Get 4h data for Williams Alligator
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Williams Alligator (Smoothed Moving Average - SMMA)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    
    # Williams Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # Using SMMA (Smoothed Moving Average) which is similar to RMA/Wilder's MA
    def smma(source, length):
        if len(source) < length:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is simple SMA
        result[length-1] = np.mean(source[:length])
        # Subsequent values: SMMA = (Prev SMMA * (length-1) + Current) / length
        for i in range(length, len(source)):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    jaw = smma(typical_price_4h, 13)  # Blue line
    teeth = smma(typical_price_4h, 8)   # Red line
    lips = smma(typical_price_4h, 5)    # Green line
    
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
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
    start_idx = max(13, 8, 5, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
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
            
            # Long: price crosses above Alligator (Lips > Teeth) AND 1d trend bullish AND volume confirmed
            # Alligator sleeping: Jaw > Teeth > Lips (all converged) - wait for awakening
            # Alligator awakening: Lips crosses above Teeth (for long) or below Teeth (for short)
            if lips_aligned[i] > teeth_aligned[i] and trend_bullish and vol_confirmed:
                # Additional filter: ensure we're not in extremely overbought conditions
                # Avoid buying when price is significantly above all lines
                if curr_close < (jaw_aligned[i] * 1.05):  # Not more than 5% above Jaw
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            # Short: price crosses below Alligator (Lips < Teeth) AND 1d trend bearish AND volume confirmed
            elif lips_aligned[i] < teeth_aligned[i] and trend_bearish and vol_confirmed:
                # Additional filter: ensure we're not in extremely oversold conditions
                # Avoid selling when price is significantly below all lines
                if curr_close > (jaw_aligned[i] * 0.95):  # Not more than 5% below Jaw
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price crosses below Teeth (reversal signal)
            stop_loss = entry_price - 2.0 * atr_4h_aligned[i]
            if curr_low < stop_loss or lips_aligned[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price crosses above Teeth (reversal signal)
            stop_loss = entry_price + 2.0 * atr_4h_aligned[i]
            if curr_high > stop_loss or lips_aligned[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0