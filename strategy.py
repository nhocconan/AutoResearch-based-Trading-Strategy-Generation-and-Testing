#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Price breaks above/below 12h Williams Alligator Jaw (SMMA13) with volume > 2.0 * 20-period volume MA and 1d EMA50 alignment.
- Exit: ATR-based stoploss (2.5 * ATR(14)) or Alligator signal reversal (price crosses opposite Alligator line).
- Signal size: 0.25 discrete to balance capture and fee control.
Williams Alligator (Jaw=SMMA13, Teeth=SMMA8, Lips=SMMA5) identifies trend absence (all lines intertwined) vs presence (lines diverged). In trending markets, price stays above/below Jaw. Works in both bull/bear by following 1d trend while using 12h Alligator for entry timing and volatility expansion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=np.float64)
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple SMA
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Williams Alligator (Jaw=SMMA13, Teeth=SMMA8, Lips=SMMA5)
    close_12h = df_12h['close'].values
    jaw = smma(close_12h, 13)   # SMMA(13)
    teeth = smma(close_12h, 8)  # SMMA(8)
    lips = smma(close_12h, 5)   # SMMA(5)
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 12h ATR(14) for stoploss
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h volume MA(20) for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(13, 50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
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
            vol_confirmed = curr_volume > 2.0 * vol_ma_12h_aligned[i]
            
            # Determine 1d EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
            trend_bullish = close[i] > ema_50_aligned[i]
            trend_bearish = close[i] < ema_50_aligned[i]
            
            # Long: price breaks above Alligator Jaw AND 1d trend bullish AND volume confirmed
            if curr_high > jaw_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Alligator Jaw AND 1d trend bearish AND volume confirmed
            elif curr_low < jaw_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Alligator Teeth (reversal signal)
            stop_loss = entry_price - 2.5 * atr_12h_aligned[i]
            if curr_low < stop_loss or curr_low < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Alligator Teeth (reversal signal)
            stop_loss = entry_price + 2.5 * atr_12h_aligned[i]
            if curr_high > stop_loss or curr_high > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Jaw_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0