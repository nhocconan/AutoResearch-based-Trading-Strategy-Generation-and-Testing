#!/usr/bin/env python3
"""
4h_1d_RSI_CCI_Pullback_V1
Hypothesis: Pullback strategy in 4h trend. Uses 1d EMA50 for trend direction, RSI(14) for oversold/overbought, CCI(20) for pullback confirmation, and volume filter. Enters long in uptrend on pullback, short in downtrend on bounce. Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for trend filter and indicators ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.replace([np.inf, -np.inf], 100).fillna(100).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d CCI(20)
    tp_1d = (high_1d + low_1d + close_1d) / 3
    sma_tp = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_1d = (tp_1d - sma_tp) / (0.015 * mad)
    cci_1d = cci_1d.fillna(0).values
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # 1d volume average (20-period) for confirmation
    vol_avg20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA50, RSI, CCI, and volume average
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(cci_1d_aligned[i]) or 
            np.isnan(vol_avg20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.3x 20-period average
        vol_filter = vol_1d_current > 1.3 * vol_avg20_1d_aligned[i]
        
        # Entry conditions: pullback in trend
        if position == 0:
            # Long: uptrend (close > EMA50), RSI oversold (<30), CCI oversold (<-100), volume
            if close[i] > ema50_1d_aligned[i] and rsi_1d_aligned[i] < 30 and cci_1d_aligned[i] < -100 and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: downtrend (close < EMA50), RSI overbought (>70), CCI overbought (>100), volume
            elif close[i] < ema50_1d_aligned[i] and rsi_1d_aligned[i] > 70 and cci_1d_aligned[i] > 100 and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse when RSI returns to neutral zone
        elif position == 1:
            if rsi_1d_aligned[i] > 50:  # exit long when RSI crosses above 50
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if rsi_1d_aligned[i] < 50:  # exit short when RSI crosses below 50
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_RSI_CCI_Pullback_V1"
timeframe = "4h"
leverage = 1.0