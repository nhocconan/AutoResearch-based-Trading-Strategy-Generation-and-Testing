#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA(50) trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA(50) for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Entry: Price breaks above/below Camarilla R3/S3 levels (calculated from prior 4h bar) with volume > 1.8 * 20-period volume MA and HTF trend alignment.
- Exit: ATR-based stoploss (1.8 * ATR(14)) or Camarilla level reversal (opposite level touch).
- Signal size: 0.25 discrete to minimize fee churn and control drawdown.
Designed to capture strong 4h momentum moves with institutional volume confirmation and trend filtering.
Works in both bull and bear markets by using 12h trend filter and volatility-based stops.
Camarilla levels provide tighter, more frequent breakout opportunities than Donchian while maintaining structure.
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
    
    # Get 4h data for Camarilla levels, ATR, and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need for volume MA
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Camarilla levels (R3, S3) from prior bar
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    # Using prior bar's high/low/close to avoid look-ahead
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    prior_high[0] = high[0]  # Fill first value
    prior_low[0] = low[0]
    prior_close[0] = close[0]
    
    camarilla_range = prior_high - prior_low
    r3 = prior_close + camarilla_range * 1.1 / 4
    s3 = prior_close - camarilla_range * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(r3[i]) or np.isnan(s3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.8x threshold)
            vol_confirmed = curr_volume > 1.8 * vol_ma[i]
            
            # Determine 12h trend: bullish if close > EMA50, bearish if close < EMA50
            htf_close_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
            htf_close = htf_close_aligned[i]
            
            trend_bullish = htf_close > ema_50_12h_aligned[i]
            trend_bearish = htf_close < ema_50_12h_aligned[i]
            
            # Long: price breaks above Camarilla R3 AND 12h trend bullish AND volume confirmed
            if curr_high > r3[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S3 AND 12h trend bearish AND volume confirmed
            elif curr_low < s3[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Camarilla S3 (reversal signal)
            stop_loss = entry_price - 1.8 * atr[i]
            if curr_low < stop_loss or curr_low < s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Camarilla R3 (reversal signal)
            stop_loss = entry_price + 1.8 * atr[i]
            if curr_high > stop_loss or curr_high > r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0