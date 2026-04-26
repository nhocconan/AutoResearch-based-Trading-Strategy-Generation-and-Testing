#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Extremes_And_Volume
Hypothesis: On 1d timeframe, Kaufman Adaptive Moving Average (KAMA) identifies the trend direction while RSI extremes (oversold <30, overbought >70) provide mean-reversion entries in the direction of the trend. Volume confirmation (>1.5x 20-day average) ensures institutional participation. This strategy works in bull markets via buying dips in uptrends and in bear markets via selling rallies in downtrends. The 1d timeframe targets 7-25 trades/year (30-100 over 4 years), minimizing fee drag while capturing significant moves. Discrete sizing (0.25) reduces churn.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA on 1d for trend direction
    # KAMA parameters: ER period=10, fast=2, slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Correct calculation of volatility (sum of absolute changes over ER period)
    er_period = 10
    volatility_sum = np.zeros(n)
    for i in range(er_period, n):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    # Avoid division by zero
    volatility_sum[volatility_sum == 0] = 1e-10
    
    # Efficiency Ratio
    er = np.zeros(n)
    er[er_period:] = np.abs(np.diff(close, prepend=close[0]))[er_period:] / volatility_sum[er_period:]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]  # seed
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    # Avoid division by zero
    avg_loss[avg_loss == 0] = 1e-10
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA, RSI, and volume MA
    start_idx = max(er_period, rsi_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        kama_val = kama[i]
        rsi_val = rsi[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: KAMA trend + RSI extreme + volume
            # Long: price > KAMA (uptrend) AND RSI < 30 (oversold) AND volume confirmed
            long_signal = (close_val > kama_val) and (rsi_val < 30) and volume_confirmed
            # Short: price < KAMA (downtrend) AND RSI > 70 (overbought) AND volume confirmed
            short_signal = (close_val < kama_val) and (rsi_val > 70) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. RSI reaches overbought (>70) in uptrend
            # 2. Price crosses below KAMA (trend change)
            if (rsi_val > 70) or (close_val < kama_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. RSI reaches oversold (<30) in downtrend
            # 2. Price crosses above KAMA (trend change)
            if (rsi_val < 30) or (close_val > kama_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Extremes_And_Volume"
timeframe = "1d"
leverage = 1.0