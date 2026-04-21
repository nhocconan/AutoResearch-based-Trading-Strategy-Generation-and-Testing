#!/usr/bin/env python3
"""
6h_Volume_Weighted_RSI_Divergence_With_Trend_Filter
Hypothesis: Combines volume-weighted RSI divergence with 1-day EMA trend filter to capture high-probability reversals in both bull and bear markets.
- In uptrend (price > daily EMA50): look for bullish RSI divergence with volume confirmation for long entries
- In downtrend (price < daily EMA50): look for bearish RSI divergence with volume confirmation for short entries
- Uses volume-weighted RSI to reduce noise and improve signal quality
- Targets 15-35 trades/year with disciplined entries to minimize fee drag
- Works in bull markets by buying dips in uptrend and in bear markets by selling rallies in downtrend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    if len(close) >= period:
        avg_gain[period-1] = np.mean(gain[:period-1])
        avg_loss[period-1] = np.mean(loss[:period-1])
        
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volume_weighted_rsi(close, volume, period=14):
    """Calculate Volume-Weighted RSI"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0) * volume[:-1]  # Volume of previous period
    loss = np.where(delta < 0, -delta, 0) * volume[:-1]
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    if len(close) >= period:
        avg_gain[period-1] = np.mean(gain[:period-1])
        avg_loss[period-1] = np.mean(loss[:period-1])
        
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema50_1d = calculate_ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Prepare data arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume-weighted RSI
    vw_rsi = calculate_volume_weighted_rsi(close, volume, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if trend filter not ready
        if np.isnan(ema50_1d_aligned[i]):
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
        
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = volume[i-20:i].mean()
            volume_ok = vol > 1.3 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Uptrend: price > daily EMA50
            if price > ema50_1d_aligned[i]:
                # Look for bullish RSI divergence: price making lower low, RSI making higher low
                if i >= 20:
                    # Find recent swing low in price
                    price_low_idx = np.argmin(close[i-20:i]) + i - 20
                    price_low = close[price_low_idx]
                    
                    # Find RSI at that point
                    rsi_at_low = vw_rsi[price_low_idx]
                    
                    # Current price and RSI
                    current_rsi = vw_rsi[i]
                    
                    # Bullish divergence: higher RSI at current low vs past low
                    if (price <= price_low * 1.005 and  # Near same price level
                        current_rsi > rsi_at_low + 10 and  # RSI significantly higher
                        volume_ok):
                        signals[i] = 0.25
                        position = 1
            # Downtrend: price < daily EMA50
            elif price < ema50_1d_aligned[i]:
                # Look for bearish RSI divergence: price making higher high, RSI making lower high
                if i >= 20:
                    # Find recent swing high in price
                    price_high_idx = np.argmax(close[i-20:i]) + i - 20
                    price_high = close[price_high_idx]
                    
                    # Find RSI at that point
                    rsi_at_high = vw_rsi[price_high_idx]
                    
                    # Current price and RSI
                    current_rsi = vw_rsi[i]
                    
                    # Bearish divergence: lower RSI at current high vs past high
                    if (price >= price_high * 0.995 and  # Near same price level
                        current_rsi < rsi_at_high - 10 and  # RSI significantly lower
                        volume_ok):
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Long exit: trend reversal or RSI overbought
            if price < ema50_1d_aligned[i] or vw_rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or RSI oversold
            if price > ema50_1d_aligned[i] or vw_rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Volume_Weighted_RSI_Divergence_With_Trend_Filter"
timeframe = "6h"
leverage = 1.0