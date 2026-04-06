#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA (Kaufman Adaptive Moving Average) with RSI filter and weekly trend.
# Uses KAMA for adaptive trend following, RSI(2) for mean reversion entries, and 1-week EMA for trend filter.
# Designed for low trade frequency (10-25/year) to minimize fee drag while capturing major moves.
# Works in bull markets via trend following and in bear markets via mean reversion against weekly trend.

name = "1d_kama_rsi_weekly_ema_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate ER (Efficiency Ratio) and SSC (Smoothing Constant)
    change = np.zeros(n)
    for i in range(1, n):
        change[i] = abs(close[i] - close[i-1])
    
    er = np.zeros(n)
    for i in range(kama_period, n):
        price_change = abs(close[i] - close[i-kama_period])
        volatility = np.sum(change[i-kama_period+1:i+1])
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # SSC = [ER * (fastest SC - slowest SC) + slowest SC]^2
    sc = np.zeros(n)
    fastest_sc = 2 / (fast_ema + 1)
    slowest_sc = 2 / (slow_ema + 1)
    for i in range(kama_period, n):
        sc[i] = (er[i] * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    if n > kama_period:
        kama[kama_period] = close[kama_period]  # Initialize with close
        for i in range(kama_period + 1, n):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(2) for mean reversion signals
    rsi_period = 2
    rsi = np.full(n, np.nan)
    if n >= rsi_period + 1:
        change_up = np.zeros(n)
        change_down = np.zeros(n)
        for i in range(1, n):
            change = close[i] - close[i-1]
            if change > 0:
                change_up[i] = change
            else:
                change_down[i] = -change
        
        # Wilder's smoothing
        up_sum = np.zeros(n)
        down_sum = np.zeros(n)
        if n >= rsi_period:
            up_sum[rsi_period-1] = np.sum(change_up[1:rsi_period+1])
            down_sum[rsi_period-1] = np.sum(change_down[1:rsi_period+1])
            for i in range(rsi_period, n):
                up_sum[i] = (up_sum[i-1] * (rsi_period-1) + change_up[i]) / rsi_period
                down_sum[i] = (down_sum[i-1] * (rsi_period-1) + change_down[i]) / rsi_period
        
        rs = np.zeros(n)
        for i in range(rsi_period, n):
            if down_sum[i] != 0:
                rs[i] = up_sum[i] / down_sum[i]
            else:
                rs[i] = 100  # Avoid division by zero
        
        for i in range(rsi_period, n):
            if rs[i] != 0:
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100
    
    # 1-week EMA for trend filter (using weekly data)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema_1w_period = 20
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_1w_period:
        ema_1w[ema_1w_period-1] = np.mean(close_1w[:ema_1w_period])
        for i in range(ema_1w_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * (ema_1w_period - 1)) / (ema_1w_period + 1)
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(kama_period + 2, rsi_period + 2, ema_1w_period + 2, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below KAMA or stoploss hit
            if (close[i] < kama[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above KAMA or stoploss hit
            if (close[i] > kama[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: RSI oversold (<15) with volume and price above weekly EMA (bullish bias)
            if (rsi[i] < 15 and volume_filter and 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: RSI overbought (>85) with volume and price below weekly EMA (bearish bias)
            elif (rsi[i] > 85 and volume_filter and 
                  close[i] < ema_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals