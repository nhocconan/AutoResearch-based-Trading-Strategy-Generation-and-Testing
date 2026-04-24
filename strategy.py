#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme + 1d EMA50 trend filter with volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Williams %R(14) crosses above -80 from below (oversold bounce) in bullish trend OR crosses below -20 from above (overbought rejection) in bearish trend, with volume > 2.0 * 20-period volume MA.
- Exit: Williams %R crosses above -20 (long exit) or below -80 (short exit) or ATR-based stoploss (2.0 * ATR(14)).
- Signal size: 0.25 discrete to balance capture and fee control.
Williams %R captures mean reversion extremes while 1d EMA50 ensures alignment with higher timeframe trend.
Volume spike filter reduces false signals in low momentum environments.
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
    
    # Get 6h data for Williams %R and ATR
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 6h Williams %R(14)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe (already on 6h, but ensure proper alignment)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Calculate 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
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
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr_6h_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
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
            
            # Determine 1d EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
            trend_bullish = close[i] > ema_50_aligned[i]
            trend_bearish = close[i] < ema_50_aligned[i]
            
            # Long: Williams %R crosses above -80 from below (oversold bounce) AND 1d trend bullish AND volume confirmed
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                trend_bullish and vol_confirmed):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R crosses below -20 from above (overbought rejection) AND 1d trend bearish AND volume confirmed
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  trend_bearish and vol_confirmed):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on Williams %R crossing above -20 (overbought) or ATR stoploss
            stop_loss = entry_price - 2.0 * atr_6h_aligned[i]
            if williams_r_aligned[i] > -20 or curr_low < stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on Williams %R crossing below -80 (oversold) or ATR stoploss
            stop_loss = entry_price + 2.0 * atr_6h_aligned[i]
            if williams_r_aligned[i] < -80 or curr_high > stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_extreme_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0