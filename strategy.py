#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout + 1w EMA(50) trend filter + volume confirmation + ATR stoploss.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA(50) for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Camarilla levels: calculated from prior 1d OHLC; long on break above R3, short on breakdown below S3.
- Volume confirmation: current volume > 2.0 * 20-period volume MA to filter weak signals.
- ATR-based stoploss: exit when price moves against position by 2.0 * ATR(14) (using 1d ATR).
- Signal size: 0.30 discrete to balance return and drawdown control.
Designed to capture strong daily moves with proper filtering to avoid overtrading and fee drag.
Works in both bull and bear markets by using 1w trend filter and volatility-based stops.
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
    
    # Get 1d data for ATR and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need for volume MA and ATR
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior 1d OHLC
    h1 = df_1d['high'].values
    l1 = df_1d['low'].values
    c1 = df_1d['close'].values
    camarilla_range = h1 - l1
    camarilla_r3 = c1 + camarilla_range * 1.1 / 4
    camarilla_s3 = c1 - camarilla_range * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
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
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Determine 1w trend: bullish if close > EMA50, bearish if close < EMA50
            htf_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
            htf_close = htf_close_aligned[i]
            
            trend_bullish = htf_close > ema_50_1w_aligned[i]
            trend_bearish = htf_close < ema_50_1w_aligned[i]
            
            # Long: price breaks above Camarilla R3 AND 1w trend bullish AND volume confirmed
            if curr_high > camarilla_r3_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S3 AND 1w trend bearish AND volume confirmed
            elif curr_low < camarilla_s3_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Camarilla S3 (reversal signal)
            stop_loss = entry_price - 2.0 * atr[i]
            if curr_low < stop_loss or curr_low < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Camarilla R3 (reversal signal)
            stop_loss = entry_price + 2.0 * atr[i]
            if curr_high > stop_loss or curr_high > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0