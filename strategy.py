#!/usr/bin/env python3
"""
6h_Daily_Camarilla_R1S1_Breakout_Volume_Strength
Hypothesis: On 6h timeframe, buy when price breaks above Camarilla R1 with volume strength in uptrend (price > daily EMA50), sell when price breaks below S1 with volume strength in downtrend (price < daily EMA50). Uses daily ATR volatility filter to avoid chop. Designed for 6h to target 20-50 trades/year with high-conviction breakout entries. Works in bull markets by capturing continuation and in bear markets by capturing breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_val = high - low
    
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    s2 = close - range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    s3 = close - range_val * 1.1 / 4
    r4 = close + range_val * 1.1 / 2
    s4 = close - range_val * 1.1 / 2
    
    return r1, s1, r2, s2, r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Camarilla, trend, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema50_1d = calculate_ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily ATR for volatility filter
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily Camarilla levels
    r1, s1, r2, s2, r3, s3, r4, s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only (avoid low-volume Asian session)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 50-period average
        if i >= 50:
            vol_ma = prices['volume'].iloc[i-50:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        if i >= 50:
            vol_filter = atr_1d_aligned[i] > np.percentile(atr_1d_aligned[:i+1], 30)
        else:
            vol_filter = True
        
        if position == 0:
            # Uptrend: price > daily EMA50
            if price > ema50_1d_aligned[i]:
                # Long: price breaks above R1 with volume strength
                if (price > r1_aligned[i] and 
                    volume_ok and vol_filter):
                    signals[i] = 0.25
                    position = 1
            # Downtrend: price < daily EMA50
            elif price < ema50_1d_aligned[i]:
                # Short: price breaks below S1 with volume strength
                if (price < s1_aligned[i] and 
                    volume_ok and vol_filter):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: trend reversal or volatility drops
            if price < ema50_1d_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or volatility drops
            if price > ema50_1d_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Daily_Camarilla_R1S1_Breakout_Volume_Strength"
timeframe = "6h"
leverage = 1.0