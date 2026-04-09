#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + Choppiness Index regime filter
# In trending regimes (CHOP < 38.2): follow KAMA direction (price > KAMA = long, price < KAMA = short)
# In ranging regimes (CHOP > 61.8): mean reversion at RSI extremes (RSI < 30 = long, RSI > 70 = short)
# Uses discrete position sizing 0.25 to limit trades to ~7-25/year and reduce fee drag
# Works in bull/bear markets: trend following captures moves, chop filter avoids whipsaws in ranging markets

name = "1d_kama_rsi_chop_v1"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else np.array([np.nan])
        # For vectorized calculation, we need to compute volatility differently
        er = np.full(len(close), np.nan)
        for i in range(period, len(close)):
            price_change = np.abs(close[i] - close[i-period])
            price_volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
            if price_volatility > 0:
                er[i] = price_change / price_volatility
            else:
                er[i] = 1.0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama = np.full(len(close), np.nan)
        kama[period] = close[period]  # Initialize
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # Calculate RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(len(close), np.nan)
        avg_loss = np.full(len(close), np.nan)
        # Wilder's smoothing
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            for i in range(period, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Load 1w data ONCE before loop for Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ATR(14) for Choppiness Index
    def calculate_atr(high, low, close, period=14):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = np.full(len(close), np.nan)
        for i in range(period, len(tr)):
            if i == period:
                atr[i] = np.nanmean(tr[1:i+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, period=14)
    
    # Calculate 1w Choppiness Index (CHOP)
    hh_1w = np.full(len(high_1w), np.nan)
    ll_1w = np.full(len(low_1w), np.nan)
    for i in range(14, len(high_1w)):
        hh_1w[i] = np.max(high_1w[i-13:i+1])
        ll_1w[i] = np.min(low_1w[i-13:i+1])
    
    sum_atr_14 = np.full(len(atr_1w), np.nan)
    for i in range(14, len(atr_1w)):
        sum_atr_14[i] = np.sum(atr_1w[i-13:i+1])
    
    range_14 = hh_1w - ll_1w
    chop_1w = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Align 1w indicators to 1d timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on weekly chop
        trending_regime = chop_1w_aligned[i] < 38.2
        ranging_regime = chop_1w_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price falls below KAMA or we enter ranging regime
                if close[i] < kama[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if RSI rises above 50 (mean reversion complete)
                if rsi[i] > 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price rises above KAMA or we enter ranging regime
                if close[i] > kama[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if RSI falls below 50 (mean reversion complete)
                if rsi[i] < 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long on price above KAMA
                if close[i] > kama[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on price below KAMA
                elif close[i] < kama[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy at RSI < 30, sell at RSI > 70
                if rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals