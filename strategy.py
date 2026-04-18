#!/usr/bin/env python3
"""
1d_1day_KAMA_Trend_RSI_With_Chop_Filter
Hypothesis: KAMA trend direction combined with RSI momentum and Choppiness index regime filter 
captures trending moves while avoiding choppy markets. Works in both bull and bear by 
adapting to market regime. Target: 15-25 trades/year (60-100 total over 4 years).
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
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    def calculate_kama(close_prices, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.abs(np.diff(close_prices))
        
        er_num = np.abs(close_prices - np.roll(close_prices, er_length))
        er_den = np.sum(volatility.reshape(-1, 1) * np.triu(np.ones((er_length, er_length))), axis=1)
        er = np.where(er_den != 0, er_num / er_den, 0)
        
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close_prices)
        kama[0] = close_prices[0]
        
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        
        return kama
    
    # Choppiness Index - regime filter
    def calculate_choppiness(high_prices, low_prices, close_prices, period=14):
        atr = np.zeros_like(close_prices)
        tr1 = np.abs(high_prices - low_prices)
        tr2 = np.abs(high_prices - np.roll(close_prices, 1))
        tr3 = np.abs(low_prices - np.roll(close_prices, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR calculation
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Sum of ATR over period
        sum_atr = np.zeros_like(close_prices)
        for i in range(period-1, len(close_prices)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        
        # High-Low range over period
        max_high = np.zeros_like(close_prices)
        min_low = np.zeros_like(close_prices)
        for i in range(period-1, len(close_prices)):
            max_high[i] = np.max(high_prices[i-period+1:i+1])
            min_low[i] = np.min(low_prices[i-period+1:i+1])
        
        # Choppiness formula
        range_hl = max_high - min_low
        cpi = np.zeros_like(close_prices)
        for i in range(period-1, len(close_prices)):
            if sum_atr[i] > 0 and range_hl[i] > 0:
                cpi[i] = 100 * np.log10(sum_atr[i] / range_hl[i]) / np.log10(period)
            else:
                cpi[i] = 50  # neutral
        
        return cpi
    
    # Calculate indicators
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    rsi_input = pd.Series(close)
    delta = rsi_input.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        weekly_trend = ema_1w_aligned[i]
        
        # Regime filter: chop < 61.8 = trending (favor trend following)
        # chop > 61.8 = choppy (favor mean reversion or stay out)
        trending_regime = chop_val < 61.8
        
        if position == 0:
            # Long: KAMA up, RSI > 50, in trending regime, above weekly trend
            if price > kama_val and rsi_val > 50 and trending_regime and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, in trending regime, below weekly trend
            elif price < kama_val and rsi_val < 50 and trending_regime and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA or RSI < 40 or choppy regime
            if price < kama_val or rsi_val < 40 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA or RSI > 60 or choppy regime
            if price > kama_val or rsi_val > 60 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1day_KAMA_Trend_RSI_With_Chop_Filter"
timeframe = "1d"
leverage = 1.0