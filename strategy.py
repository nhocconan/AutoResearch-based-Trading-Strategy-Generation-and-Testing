#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA (adaptive moving average) with weekly RSI filter and volume confirmation.
# Long when price crosses above KAMA with weekly RSI < 50 (avoiding overbought) and volume > 1.5x average.
# Short when price crosses below KAMA with weekly RSI > 50 (avoiding oversold) and volume > 1.5x average.
# Exit when price crosses back through KAMA.
# KAMA adapts to market noise, reducing whipsaws in choppy markets. Weekly RSI avoids trend exhaustion.
# Target: 10-25 trades per year on 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI (14-period)
    def calculate_rsi(prices, period):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(prices, np.nan)
        avg_loss = np.full_like(prices, np.nan)
        
        if len(prices) >= period + 1:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            for i in range(period + 1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1w = calculate_rsi(close_1w, 14)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate KAMA (adaptive moving average) on daily close
    def calculate_kama(prices, er_period=10, fast_sc=2, slow_sc=30):
        if len(prices) < er_period:
            return np.full_like(prices, np.nan)
        
        change = np.abs(np.diff(prices, er_period))
        volatility = np.sum(np.abs(np.diff(prices)), axis=0) if len(prices) > 1 else 0
        
        # Vectorized calculation
        er = np.zeros_like(prices)
        for i in range(er_period, len(prices)):
            if volatility[i-er_period:i].sum() > 0:
                er[i] = change[i] / volatility[i-er_period:i].sum()
            else:
                er[i] = 0
        
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        kama = np.full_like(prices, np.nan)
        kama[er_period] = prices[er_period]
        
        for i in range(er_period + 1, len(prices)):
            kama[i] = kama[i-1] + sc[i] * (prices[i] - kama[i-1])
        
        return kama
    
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA, weekly RSI, and volume MA20
    start_idx = max(30, 20)  # KAMA needs ~30 bars, volume MA20 needs 19
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi_1w_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price crosses above KAMA with weekly RSI < 50 and volume filter
            if (close[i-1] <= kama[i-1] and price > kama_val and 
                rsi_val < 50 and vol_filter):
                signals[i] = size
                position = 1
            # Short: price crosses below KAMA with weekly RSI > 50 and volume filter
            elif (close[i-1] >= kama[i-1] and price < kama_val and 
                  rsi_val > 50 and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i-1] >= kama[i-1] and price < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i-1] <= kama[i-1] and price > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_WklyRSI_Volume"
timeframe = "1d"
leverage = 1.0