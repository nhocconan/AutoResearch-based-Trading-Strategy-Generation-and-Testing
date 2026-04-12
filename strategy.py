#!/usr/bin/env python3
"""
1h_4h_1d_rsi_vwap_momentum
Hypothesis: 1-hour strategy using VWAP pullbacks with RSI momentum in the direction of 4h/1d trend.
Combines VWAP mean reversion with RSI momentum for entries, filtered by higher timeframe trend.
Designed for low trade frequency (15-30/year) to minimize fee drag in 1h timeframe.
Uses 4h for trend direction and 1d for volatility filter to avoid choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend direction
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR for volatility filter (avoid choppy markets)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.roll(high_1d, 1) - close_1d)
    tr3 = np.abs(np.roll(low_1d, 1) - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, typical_price)
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility (chop) and extremely high volatility
        atr_median = np.nanmedian(atr_1d_aligned[max(0, i-50):i+1])
        if atr_1d_aligned[i] < atr_median * 0.5 or atr_1d_aligned[i] > atr_median * 3.0:
            # Hold current position or stay flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Trend determination from 4h EMA50
        uptrend = ema50_4h_aligned[i] > close[i]  # Price above 4h EMA50 = uptrend bias
        downtrend = ema50_4h_aligned[i] < close[i]  # Price below 4h EMA50 = downtrend bias
        
        # VWAP deviation for mean reversion signals
        vwap_dev = (close[i] - vwap[i]) / vwap[i]
        
        # Entry conditions: VWAP pullback with RSI momentum in trend direction
        if uptrend and position != 1:
            # Long: price below VWAP (pullback) with RSI showing momentum
            if vwap_dev < -0.001 and rsi[i] > 50 and rsi[i] < 70:
                position = 1
                signals[i] = 0.20
        elif downtrend and position != -1:
            # Short: price above VWAP (pullback) with RSI showing momentum
            if vwap_dev > 0.001 and rsi[i] < 50 and rsi[i] > 30:
                position = -1
                signals[i] = -0.20
        # Exit conditions: VWAP mean reversion or RSI extreme
        elif position == 1 and (vwap_dev > 0.001 or rsi[i] > 70):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (vwap_dev < -0.001 or rsi[i] < 30):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_rsi_vwap_momentum"
timeframe = "1h"
leverage = 1.0