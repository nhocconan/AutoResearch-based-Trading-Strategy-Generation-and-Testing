#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA direction with weekly RSI filter and volatility-adjusted position sizing
# Uses 1d KAMA for trend direction, 1w RSI for overbought/oversold extremes, and ATR-based sizing
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag
# Works in bull markets (trend following) and bear markets (mean reversion via RSI extremes)

name = "1d_kama_rsi_weekly_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for KAMA calculation
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Handle first 10 elements
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility[1:]])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1-week data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # RSI calculation with Wilder's smoothing
    avg_gain = np.full(len(close_1w), np.nan)
    avg_loss = np.full(len(close_1w), np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.concatenate([np.full(14, np.nan), rsi_1w[14:]])
    
    # Align 1w RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # ATR(14) for volatility assessment and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI extremes: oversold < 30, overbought > 70
        rsi_oversold = rsi_1w_aligned[i] < 30
        rsi_overbought = rsi_1w_aligned[i] > 70
        
        # Volatility normalization: size inversely proportional to volatility
        # Base size 0.25, scaled by ATR relative to 50-period average
        if i >= 50:
            atr_avg = np.nanmean(atr[i-50:i])
            if not np.isnan(atr_avg) and atr_avg > 0:
                vol_scale = min(2.0, max(0.5, atr_avg / atr[i]))
            else:
                vol_scale = 1.0
        else:
            vol_scale = 1.0
        
        base_size = 0.25
        size = base_size * vol_scale
        
        # Long conditions: price above KAMA AND RSI oversold (trend + mean reversion)
        if price_above_kama and rsi_oversold:
            signals[i] = size
        # Short conditions: price below KAMA AND RSI overbought (trend + mean reversion)
        elif price_below_kama and rsi_overbought:
            signals[i] = -size
        else:
            signals[i] = 0.0
    
    return signals