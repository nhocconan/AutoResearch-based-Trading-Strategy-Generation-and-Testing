#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout + 1d EMA(34) trend filter + volume confirmation + ATR stoploss.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA(34) for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Camarilla levels: calculated from prior 12h OHLC; long on break above H3, short on breakdown below L3.
- Volume confirmation: current volume > 2.0 * 20-period volume MA to filter weak signals.
- ATR-based stoploss: exit when price moves against position by 2.5 * ATR(14) (using 12h ATR).
- Signal size: 0.25 discrete to reduce fee churn and control drawdown.
Designed to capture strong 12h moves with proper filtering to avoid overtrading and fee drag.
Works in both bull and bear markets by using 1d trend filter and volatility-based stops.
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
    
    # Get 12h data for ATR and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need for volume MA and ATR
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior 12h OHLC
    h1 = df_12h['high'].values
    l1 = df_12h['low'].values
    c1 = df_12h['close'].values
    camarilla_range = h1 - l1
    camarilla_h3 = c1 + camarilla_range * 1.1 / 4
    camarilla_l3 = c1 - camarilla_range * 1.1 / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
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
            
            # Determine 1d trend: bullish if close > EMA34, bearish if close < EMA34
            htf_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
            htf_close = htf_close_aligned[i]
            
            trend_bullish = htf_close > ema_34_1d_aligned[i]
            trend_bearish = htf_close < ema_34_1d_aligned[i]
            
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
            stop_loss = entry_price - 2.5 * atr[i]
            if curr_low < stop_loss or curr_low < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Camarilla H3 (reversal signal)
            stop_loss = entry_price + 2.5 * atr[i]
            if curr_high > stop_loss or curr_high > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0