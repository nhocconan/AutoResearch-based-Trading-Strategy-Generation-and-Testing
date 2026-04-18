# 1d KAMA + RSI + Chop Regime Strategy
# Hypothesis: KAMA adapts to market noise, reducing whipsaws. Combined with RSI extremes
# and chop regime filter, it captures trends while avoiding sideways chop. Works in bull
# and bear markets by following adaptive trend direction with momentum confirmation.
# Timeframe: 1d (lower trade frequency for better generalization)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average"""
    if len(close) < er_length:
        return np.full_like(close, np.nan)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # Vectorized volatility calculation
    volatility = np.array([np.sum(np.abs(np.diff(close[max(0, i-er_length+1):i+1]))) 
                          for i in range(len(close))])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index: 0-100, >61.8 = choppy, <38.2 = trending"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    
    # Sum of True Range over period
    sum_tr = np.zeros_like(high)
    for i in range(len(sum_tr)):
        if i < period:
            sum_tr[i] = np.sum(tr[max(0, i-period+1):i+1])
        else:
            sum_tr[i] = np.sum(tr[i-period+1:i+1])
    
    # Highest high and lowest low over period
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    for i in range(len(high_high)):
        if i < period:
            highest_high[i] = np.max(high[max(0, i-period+1):i+1])
            lowest_low[i] = np.min(low[max(0, i-period+1):i+1])
        else:
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    # Choppiness formula
    chop = 100 * np.log10(sum_tr / range_hl) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate KAMA on weekly close for trend direction
    wk_close = df_1w['close'].values
    wk_kama = calculate_kama(wk_close, er_length=10, fast_sc=2, slow_sc=30)
    wk_kama_trend = wk_kama > np.roll(wk_kama, 1)  # Rising KAMA = uptrend
    wk_kama_trend[0] = False  # First value has no previous
    wk_kama_trend_aligned = align_htf_to_ltf(prices, df_1w, wk_kama_trend.astype(float))
    
    # Calculate daily indicators
    # KAMA for entry signal
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    kama_slope = kama > np.roll(kama, 1)  # Rising KAMA
    kama_slope[0] = False
    
    # RSI for momentum confirmation
    rsi = calculate_rsi(close, period=14)
    
    # Choppiness filter for regime
    chop = calculate_choppiness(high, low, close, period=14)
    trending_market = chop < 38.2  # Strong trending regime
    
    # Volume confirmation: above average volume
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_confirm = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(wk_kama_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: KAMA rising + RSI > 50 (bullish momentum) + 
            # weekly trend up + trending market + volume confirmation
            if (kama_slope[i] and rsi[i] > 50 and 
                wk_kama_trend_aligned[i] > 0.5 and 
                trending_market[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA falling + RSI < 50 (bearish momentum) + 
            # weekly trend down + trending market + volume confirmation
            elif (not kama_slope[i] and rsi[i] < 50 and 
                  wk_kama_trend_aligned[i] < 0.5 and 
                  trending_market[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns down OR RSI < 40 (losing momentum)
            if not kama_slope[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up OR RSI > 60 (losing momentum)
            if kama_slope[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime"
timeframe = "1d"
leverage = 1.0