#!/usr/bin/env python3
"""
12h_WK1_Trend_Momentum_v1
Hypothesis: Use 1w EMA34 for trend direction and 1d RSI for momentum confirmation on 12h timeframe. 
Go long when price > 1w EMA34 AND 1d RSI > 55, short when price < 1w EMA34 AND 1d RSI < 45. 
Requires volume > 1.5x 20-period average for confirmation. Target: 15-30 trades/year by combining multiple filters to reduce noise. 
Works in bull markets via trend following and in bear via short signals. 
Multi-timeframe: 1w trend (HTF), 1d momentum (MTF), 12h execution (LTF).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34
    ema_len = 34
    ema_1w = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= ema_len:
        # Use pandas EMA for proper calculation
        ema_series = pd.Series(close_1w).ewm(span=ema_len, adjust=False).mean()
        ema_1w = ema_series.values
    
    # Align 1w EMA34 to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d RSI(14)
    rsi_period = 14
    rsi_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= rsi_period + 1:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1d, np.nan)
        avg_loss = np.full_like(close_1d, np.nan)
        
        # First average
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Time-based filters
    hours = pd.DatetimeIndex(prices['open_time']).hour
    # Avoid low liquidity hours: 22-06 UTC (6 hours)
    # Active hours: 06-22 UTC (16 hours)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_len, rsi_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Time filter: avoid 22-06 UTC (low liquidity)
        hour = hours[i]
        in_active_hours = 6 <= hour < 22
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and in_active_hours:
            # Long: price > 1w EMA34 AND 1d RSI > 55 AND volume confirmation
            if close[i] > ema_1w_aligned[i] and rsi_1d_aligned[i] > 55 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price < 1w EMA34 AND 1d RSI < 45 AND volume confirmation
            elif close[i] < ema_1w_aligned[i] and rsi_1d_aligned[i] < 45 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < 1w EMA34 OR 1d RSI < 40
            if close[i] < ema_1w_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > 1w EMA34 OR 1d RSI > 60
            if close[i] > ema_1w_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WK1_Trend_Momentum_v1"
timeframe = "12h"
leverage = 1.0