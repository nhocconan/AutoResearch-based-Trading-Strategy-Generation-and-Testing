#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Volume_and_Chop_Filter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction on 12h timeframe.
Enter long when price > KAMA and volume spike with choppy market filter.
Enter short when price < KAMA and volume spike with choppy market filter.
Use daily ATR for volatility filter and daily EMA200 for additional trend confirmation.
Designed for 12h timeframe to target 20-30 trades/year with high-conviction entries.
KAMA adapts to market noise, reducing whipsaws in choppy markets while capturing trends.
Works in bull markets by capturing uptrends and in bear markets by capturing downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_ema=2, slow_ema=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # For efficiency, compute volatility as rolling sum of absolute changes
    volatility = np.convolve(np.abs(np.diff(close, prepend=close[0])), np.ones(er_period), mode='same')
    volatility[:er_period-1] = np.nan
    
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    kama = np.full_like(close, np.nan)
    if len(close) > 0:
        kama[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    return kama

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.full_like(close, np.nan)
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
    
    atr = np.full_like(tr, np.nan)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend, volatility, and chop filter
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
    
    # Calculate KAMA on 12h price
    kama = calculate_kama(prices['close'].values, er_period=10, fast_ema=2, slow_ema=30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(kama[i])):
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
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        if i >= 20:
            vol_filter = atr_1d_aligned[i] > np.percentile(atr_1d_aligned[max(0,i-20):i+1], 30)
        else:
            vol_filter = True
        
        if position == 0:
            # Uptrend: price > KAMA and price > daily EMA200
            if price > kama[i] and price > ema200_1d_aligned[i]:
                # Long: price above KAMA with volume confirmation
                if volume_ok and vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Downtrend: price < KAMA and price < daily EMA200
            elif price < kama[i] and price < ema200_1d_aligned[i]:
                # Short: price below KAMA with volume confirmation
                if volume_ok and vol_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: trend reversal or volatility drops
            if price < kama[i] or price < ema200_1d_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or volatility drops
            if price > kama[i] or price > ema200_1d_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_With_Volume_and_Chop_Filter"
timeframe = "12h"
leverage = 1.0