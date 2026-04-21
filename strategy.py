#!/usr/bin/env python3
"""
12h_Daily_Williams_Fractal_Trend_Follow
Hypothesis: Follow daily trend using Williams fractal breakouts on 12h timeframe. 
In uptrend (price > daily EMA200), buy when price breaks above bearish fractal with volume confirmation.
In downtrend (price < daily EMA200), sell when price breaks below bullish fractal with volume confirmation.
Uses daily ATR for volatility filter to avoid choppy markets.
Designed for 12h timeframe to target 15-35 trades/year with high-conviction entries.
Works in bull markets by capturing continuation and in bear markets by capturing breakdowns.
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

def calculate_williams_fractals(high, low):
    """Calculate Williams fractals: bearish (high point) and bullish (low point)"""
    n = len(high)
    bearish = np.zeros(n)
    bullish = np.zeros(n)
    
    for i in range(2, n-2):
        # Bearish fractal: highest high in middle
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: lowest low in middle
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend, fractals, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend filter
    ema200_1d = calculate_ema(close_1d, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Daily ATR for volatility filter
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Williams fractals on daily (need 2-bar confirmation)
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
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
        
        # Volume filter: current volume > 1.5 * 50-period average
        if i >= 50:
            vol_ma = prices['volume'].iloc[i-50:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        if i >= 50:
            vol_filter = atr_1d_aligned[i] > np.percentile(atr_1d_aligned[:i+1], 30)
        else:
            vol_filter = True
        
        if position == 0:
            # Uptrend: price > daily EMA200
            if price > ema200_1d_aligned[i]:
                # Long: price breaks above bearish fractal with volume confirmation
                if (price > bearish_fractal_aligned[i] and 
                    volume_ok and vol_filter):
                    signals[i] = 0.25
                    position = 1
            # Downtrend: price < daily EMA200
            elif price < ema200_1d_aligned[i]:
                # Short: price breaks below bullish fractal with volume confirmation
                if (price < bullish_fractal_aligned[i] and 
                    volume_ok and vol_filter):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: trend reversal or volatility drops
            if price < ema200_1d_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or volatility drops
            if price > ema200_1d_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Daily_Williams_Fractal_Trend_Follow"
timeframe = "12h"
leverage = 1.0