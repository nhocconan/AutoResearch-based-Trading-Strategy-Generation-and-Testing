#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volume-Weighted Average Price (VWAP) deviation + 1d EMA50 trend filter + ATR-based volatility filter
# Long when price > VWAP (bullish bias) AND 1d uptrend AND volatility expansion (ATR ratio > 1.0)
# Short when price < VWAP (bearish bias) AND 1d downtrend AND volatility expansion (ATR ratio > 1.0)
# VWAP provides dynamic support/resistance; EMA50 filters counter-trend trades; ATR ratio ensures trades occur during volatile regimes
# Designed for 30-60 trades/year on 4h to minimize fee drag while capturing volatile breakouts in both bull and bear markets

name = "4h_VWAP_Deviation_1dEMA50_ATR_VolFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, 0.0)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar TR
    tr2[0] = np.abs(high[0] - close[0])  # First bar TR
    tr3[0] = np.abs(low[0] - close[0])   # First bar TR
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr / atr_ma_50  # Current ATR vs 50-period average ATR
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(atr_ratio[i]) or vwap_den[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > VWAP (bullish bias) AND 1d uptrend AND volatility expansion
            if (close[i] > vwap[i] and 
                close[i] > ema_50_aligned[i] and  # 1d uptrend
                atr_ratio[i] > 1.0):  # Volatility expansion
                signals[i] = 0.25
                position = 1
            # Short conditions: price < VWAP (bearish bias) AND 1d downtrend AND volatility expansion
            elif (close[i] < vwap[i] and 
                  close[i] < ema_50_aligned[i] and  # 1d downtrend
                  atr_ratio[i] > 1.0):  # Volatility expansion
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < VWAP OR 1d trend turns down OR volatility contraction
            if (close[i] < vwap[i] or 
                close[i] < ema_50_aligned[i] or 
                atr_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > VWAP OR 1d trend turns up OR volatility contraction
            if (close[i] > vwap[i] or 
                close[i] > ema_50_aligned[i] or 
                atr_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals