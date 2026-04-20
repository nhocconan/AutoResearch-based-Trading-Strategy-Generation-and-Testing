#!/usr/bin/env python3
"""
Hypothesis: 12h strategy combining 1-week ADX for trend strength and 1-day RSI for mean-reversion entries.
Long when weekly ADX > 25 (trending) and daily RSI < 30 (oversold pullback).
Short when weekly ADX > 25 and daily RSI > 70 (overbought rally).
Exit when RSI returns to neutral (40-60) or ADX weakens (<20).
Uses volume confirmation to avoid false signals.
Designed for 12h timeframe with ~15-25 trades/year, avoiding overtrading.
Works in bull/bear by trading pullbacks in strong trends only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], 
                                   np.abs(high_1w[0] - close_1w[0]),
                                   np.abs(low_1w[0] - close_1w[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smooth(tr, 14)
    plus_di_1w = 100 * wilders_smooth(dm_plus, 14) / (atr_1w + 1e-10)
    minus_di_1w = 100 * wilders_smooth(dm_minus, 14) / (atr_1w + 1e-10)
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
    adx_1w = wilders_smooth(dx_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate daily volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).ewm(span=20, adjust=False).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        adx_val = adx_1w_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_val = volume_1d[i]
        vol_avg_val = vol_avg_20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(rsi_val) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: strong trend (ADX > 25) + oversold (RSI < 30) + volume confirmation
            if adx_val > 25 and rsi_val < 30 and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: strong trend (ADX > 25) + overbought (RSI > 70) + volume confirmation
            elif adx_val > 25 and rsi_val > 70 and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (>40) or trend weakens (ADX < 20)
            if rsi_val > 40 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (<60) or trend weakens (ADX < 20)
            if rsi_val < 60 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_ADX_RSI_Volume
# Uses 1-week ADX for trend strength and 1-day RSI for mean-reversion entries
# Enters long when weekly ADX > 25 (strong trend) and daily RSI < 30 (oversold)
# Enters short when weekly ADX > 25 and daily RSI > 70 (overbought)
# Exits when RSI returns to neutral range (40-60 for long, 60-40 for short) or ADX weakens (<20)
# Includes volume confirmation to avoid false signals
# Designed for 12h timeframe with ~15-25 trades/year
name = "12h_ADX_RSI_Volume"
timeframe = "12h"
leverage = 1.0