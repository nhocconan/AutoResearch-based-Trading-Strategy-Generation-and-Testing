#!/usr/bin/env python3
"""
1h_4hSupertrend_1dRSI_Filter
Hypothesis: Use 4h Supertrend for trend direction (works in bull/bear by adapting to volatility) and 1d RSI for pullback entries. Enter long when 4h Supertrend is bullish and 1d RSI < 40 (pullback in uptrend), short when 4h Supertrend is bearish and 1d RSI > 60 (pullback in downtrend). Use 1h timeframe only for entry timing to avoid whipsaw. Apply session filter (08-20 UTC) to reduce noise. Target 15-35 trades/year via tight RSI thresholds and Supertrend's trend-filtering. Works in ranging markets by avoiding entries when RSI is neutral (40-60). Uses volume > 1.5x 24-period average for confirmation to avoid low-volume breakouts.
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
    
    # Get 4h data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Supertrend on 4h (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.max([high_4h[0] - low_4h[0], np.abs(high_4h[0] - close_4h[0]), np.abs(low_4h[0] - close_4h[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing
    atr = np.full_like(close_4h, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    upper_band = np.full_like(close_4h, np.nan)
    lower_band = np.full_like(close_4h, np.nan)
    supertrend = np.full_like(close_4h, np.nan)
    trend = np.full_like(close_4h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_4h)):
        if np.isnan(atr[i]):
            continue
        upper_band[i] = (high_4h[i] + low_4h[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high_4h[i] + low_4h[i]) / 2 - multiplier * atr[i]
        
        if i == 0:
            supertrend[i] = upper_band[i]
            trend[i] = 1
        else:
            if close_4h[i-1] > upper_band[i-1]:
                trend[i] = 1
            elif close_4h[i-1] < lower_band[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
            
            if trend[i] == 1:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d RSI(14) with proper Wilder smoothing
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
    
    # Align 4h Supertrend trend and 1d RSI to 1h timeframe
    trend_aligned = align_htf_to_ltf(prices, df_4h, trend)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trend_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h Supertrend bullish (trend=1) + 1d RSI < 40 (pullback) + volume
            if trend_aligned[i] == 1 and rsi_1d_aligned[i] < 40 and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: 4h Supertrend bearish (trend=-1) + 1d RSI > 60 (pullback) + volume
            elif trend_aligned[i] == -1 and rsi_1d_aligned[i] > 60 and vol_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h Supertrend turns bearish or 1d RSI > 60
            if trend_aligned[i] == -1 or rsi_1d_aligned[i] > 60:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h Supertrend turns bullish or 1d RSI < 40
            if trend_aligned[i] == 1 or rsi_1d_aligned[i] < 40:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hSupertrend_1dRSI_Filter"
timeframe = "1h"
leverage = 1.0