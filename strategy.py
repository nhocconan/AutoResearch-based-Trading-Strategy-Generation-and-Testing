#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to determine trend direction on daily timeframe, 
combined with RSI(14) for entry timing and Choppiness Index(14) to filter ranging markets. 
Only trade long when price > KAMA and RSI < 30 in trending regimes (CHOP < 38.2), 
and short when price < KAMA and RSI > 70 in trending regimes. 
Designed for 15-25 trades/year on BTC/ETH/SOL to avoid fee drag while capturing strong trends.
Uses 1w EMA50 as higher timeframe trend filter to avoid counter-trend trades.
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
    
    # Get 1d data for KAMA, RSI, and Choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Kaufman Adaptive Moving Average (KAMA) on 1d close
    def calculate_kama(close_vals, er_len=10, fast_len=2, slow_len=30):
        kama = np.zeros_like(close_vals)
        kama[:] = np.nan
        
        if len(close_vals) < er_len:
            return kama
            
        # Efficiency Ratio
        change = np.abs(np.diff(close_vals, n=er_len))
        volatility = np.sum(np.abs(np.diff(close_vals)), axis=0)
        er = np.zeros_like(close_vals)
        er[er_len:] = change[er_len-1:] / np.where(volatility[er_len-1:] != 0, volatility[er_len-1:], 1)
        
        # Smoothing constants
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        
        # KAMA calculation
        kama[er_len] = close_vals[er_len]
        for i in range(er_len + 1, len(close_vals)):
            kama[i] = kama[i-1] + sc[i] * (close_vals[i] - kama[i-1])
            
        return kama
    
    # RSI(14) on 1d close
    def calculate_rsi(close_vals, period=14):
        rsi = np.full_like(close_vals, np.nan)
        if len(close_vals) < period + 1:
            return rsi
            
        delta = np.diff(close_vals)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_vals)
        avg_loss = np.zeros_like(close_vals)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close_vals)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
            
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index(14) on 1d OHLC
    def calculate_choppiness(high_vals, low_vals, close_vals, period=14):
        chop = np.full_like(close_vals, np.nan)
        if len(close_vals) < period:
            return chop
            
        # True Range
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_vals[0] - low_vals[0]
        
        # Sum of True Range over period
        atr_sum = np.zeros_like(close_vals)
        for i in range(period, len(close_vals)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
            
        # Highest high and lowest low over period
        hh = np.zeros_like(close_vals)
        ll = np.zeros_like(close_vals)
        for i in range(period-1, len(close_vals)):
            hh[i] = np.max(high_vals[i-period+1:i+1])
            ll[i] = np.min(low_vals[i-period+1:i+1])
            
        # Choppiness Index
        chop = 100 * np.log10(atr_sum / np.where((hh - ll) != 0, (hh - ll), 1)) / np.log10(period)
        return chop
    
    # Calculate indicators on 1d data
    kama_1d = calculate_kama(df_1d['close'].values)
    rsi_1d = calculate_rsi(df_1d['close'].values)
    chop_1d = calculate_choppiness(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # 1w EMA50 for higher timeframe trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe (prices are already 1d)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of KAMA ER length (10), RSI period (14), Choppiness period (14), EMA50 (50)
    start_idx = max(10, 14, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        kama_val = kama_1d_aligned[i]
        close_val = close[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price > KAMA, RSI < 30 (oversold), trending regime (CHOP < 38.2), and above 1w EMA50
            long_signal = (close_val > kama_val) and \
                          (rsi_val < 30) and \
                          (chop_val < 38.2) and \
                          (close_val > ema_50_1w_val)
            
            # Short: price < KAMA, RSI > 70 (overbought), trending regime (CHOP < 38.2), and below 1w EMA50
            short_signal = (close_val < kama_val) and \
                           (rsi_val > 70) and \
                           (chop_val < 38.2) and \
                           (close_val < ema_50_1w_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            # 1. Price crosses below KAMA (trend change)
            # 2. RSI > 70 (overbought exit)
            # 3. Chop >= 38.2 (ranging market)
            # 4. Price < 1w EMA50 (higher timeframe trend change)
            if (close_val < kama_val) or \
               (rsi_val > 70) or \
               (chop_val >= 38.2) or \
               (close_val < ema_50_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            # 1. Price crosses above KAMA (trend change)
            # 2. RSI < 30 (oversold exit)
            # 3. Chop >= 38.2 (ranging market)
            # 4. Price > 1w EMA50 (higher timeframe trend change)
            if (close_val > kama_val) or \
               (rsi_val < 30) or \
               (chop_val >= 38.2) or \
               (close_val > ema_50_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0