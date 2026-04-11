#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_rsi_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA calculation for daily trend
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_trend = kama > np.roll(kama, 1)  # Rising KAMA = uptrend
    kama_trend[0] = False
    
    # Daily RSI for overbought/oversold
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    # Align daily indicators to 4h
    kama_trend_4h = align_htf_to_ltf(prices, df_1d, kama_trend)
    rsi_oversold_4h = align_htf_to_ltf(prices, df_1d, rsi_oversold)
    rsi_overbought_4h = align_htf_to_ltf(prices, df_1d, rsi_overbought)
    
    # 4h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_trend_4h[i]) or np.isnan(rsi_oversold_4h[i]) or 
            np.isnan(rsi_overbought_4h[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.3x average)
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Long conditions: KAMA uptrend + RSI oversold + volume
        long_signal = volume_confirmed and kama_trend_4h[i] and rsi_oversold_4h[i]
        
        # Short conditions: KAMA downtrend + RSI overbought + volume
        short_signal = volume_confirmed and (~kama_trend_4h[i]) and rsi_overbought_4h[i]
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and (rsi_oversold_4h[i] == False and rsi_overbought_4h[i] == False)
        exit_short = position == -1 and (rsi_oversold_4h[i] == False and rsi_overbought_4h[i] == False)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily KAMA trend + RSI extremes strategy for 4h timeframe with volume confirmation.
# Uses daily KAMA to determine trend direction (rising = uptrend, falling = downtrend).
# Enters long when daily KAMA is rising AND daily RSI is oversold (<30) with volume >1.3x average.
# Enters short when daily KAMA is falling AND daily RSI is overbought (>70) with volume >1.3x average.
# Exits when RSI returns to neutral zone (40-60) to capture mean reversion within the trend.
# KAMA adapts to market noise, making it effective in both trending and ranging markets.
# RSI extremes provide high-probability reversal points in the direction of the trend.
# Volume confirmation ensures institutional participation.
# Designed for low trade frequency (~20-40 trades/year) to minimize drag.
# Works in both bull and bear markets as it follows the daily trend while fading extreme RSI readings.