#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction + RSI(2) extreme + volume confirmation
# KAMA adapts to market noise - trend following in trending markets, flat in ranging
# RSI(2) catches short-term overextensions for mean reversion entries
# Volume confirmation ensures institutional participation
# Works in bull/bear: KAMA adapts, RSI(2) captures reversals in both regimes
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25-0.30

name = "1d_kama_rsi2_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - 10 period
    def kama(close, period=10, fast=2, slow=30):
        # Calculate efficiency ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        # Vectorized ER calculation
        er = np.zeros_like(close)
        for i in range(period, len(close)):
            if volatility[i] > 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Calculate KAMA
        kama_values = np.zeros_like(close)
        kama_values[0] = close[0]
        for i in range(1, len(close)):
            kama_values[i] = kama_values[i-1] + sc[i] * (close[i] - kama_values[i-1])
        return kama_values
    
    # Calculate RSI(2)
    def rsi(close, period=2):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_values = np.zeros_like(close)
        rsi_values = 100 - (100 / (1 + rs))
        # Set first period values to 50 (neutral)
        rsi_values[:period] = 50
        return rsi_values
    
    # Calculate 1w trend filter (EMA 21)
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate indicators
    kama_values = kama(close, 10, 2, 30)
    rsi_values = rsi(close, 2)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_values[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # KAMA trend: price above/below KAMA
        price_above_kama = close[i] > kama_values[i]
        price_below_kama = close[i] < kama_values[i]
        
        # RSI(2) extremes
        rsi_oversold = rsi_values[i] < 10
        rsi_overbought = rsi_values[i] > 90
        
        if position == 1:  # Long position
            # Exit: RSI > 70 or price crosses below KAMA
            if rsi_values[i] > 70 or close[i] < kama_values[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 or price crosses above KAMA
            if rsi_values[i] < 30 or close[i] > kama_values[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if volume_confirmed:
                # Long: price above KAMA + RSI oversold + weekly uptrend
                if (price_above_kama and rsi_oversold and 
                    close[i] > ema_21_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price below KAMA + RSI overbought + weekly downtrend
                elif (price_below_kama and rsi_overbought and 
                      close[i] < ema_21_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals