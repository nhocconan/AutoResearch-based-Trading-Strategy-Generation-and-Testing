#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA with RSI and chop filter
# Uses Kaufman's Adaptive Moving Average (KAMA) to capture trend direction
# Combines with RSI for momentum and Choppiness Index for regime filtering
# Aims for 15-25 trades per year on 1d timeframe (60-100 total over 4 years)
# Works in bull markets via trend following and in bear via mean reversion in ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (2-period ER, 30-period slow, 2-period fast)
    # ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)
    er = np.zeros(n)
    er[9:] = change[9:] / volatility[9:]
    er[volatility == 0] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)      # EMA(2)
    slow_sc = 2 / (30 + 1)     # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    avg_loss[avg_loss == 0] = 1e-10
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14-period)
    # ATR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = np.zeros(n)
    for i in range(14, n):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    chop = np.zeros(n)
    denominator = highest_high - lowest_low
    denominator[denominator == 0] = 1e-10
    chop[14:] = 100 * np.log10(atr_sum[14:] / denominator[14:]) / np.log10(14)
    
    # Load 1-week data for higher timeframe filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on weekly data
    ema50_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (50 + 1)
    ema50_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema50_1w[i] = (close_1w[i] - ema50_1w[i-1]) * ema_multiplier + ema50_1w[i-1]
    
    # Align weekly EMA to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_trend = ema50_1w_aligned[i]
        
        # Chop filter: > 61.8 = ranging (mean revert), < 38.2 = trending (trend follow)
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long conditions
            if is_trending:
                # In trending markets: follow KAMA direction
                if price > kama_val and price > ema_trend and rsi_val > 50:
                    position = 1
                    signals[i] = position_size
            else:
                # In ranging markets: mean reversion from extremes
                if price < kama_val and rsi_val < 30:
                    position = 1
                    signals[i] = position_size
            
            # Short conditions
            if is_trending:
                # In trending markets: follow KAMA direction
                if price < kama_val and price < ema_trend and rsi_val < 50:
                    position = -1
                    signals[i] = -position_size
            else:
                # In ranging markets: mean reversion from extremes
                if price > kama_val and rsi_val > 70:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Exit long: reverse signals
            if is_trending:
                if price < kama_val or price < ema_trend or rsi_val < 40:
                    position = 0
                    signals[i] = 0.0
            else:
                if price > kama_val or rsi_val > 70:
                    position = 0
                    signals[i] = 0.0
        elif position == -1:
            # Exit short: reverse signals
            if is_trending:
                if price > kama_val or price > ema_trend or rsi_val > 60:
                    position = 0
                    signals[i] = 0.0
            else:
                if price < kama_val or rsi_val < 30:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "1d_1w_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0