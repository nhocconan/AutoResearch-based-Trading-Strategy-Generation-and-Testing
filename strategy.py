#!/usr/bin/env python3
# 6h_12h_1d_Momentum_Regime_Volume
# Hypothesis: Combines 12h momentum (RSI), 1d regime (ADX), and volume confirmation to capture trends while avoiding whipsaws. Works in bull via momentum breakouts in high ADX regimes, and in bear via mean reversion in low ADX regimes. Designed for low trade frequency (12-37/year) to minimize fee drag.

name = "6h_12h_1d_Momentum_Regime_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for momentum and 1d data for regime
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 14 or len(df_1d) < 14:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h RSI(14) for momentum
    rsi_period = 14
    delta = np.diff(df_12h['close'].values, prepend=df_12h['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # 1d ADX(14) for regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h ATR for volatility and stop
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = high[0] - low[0]
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_12h_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        trending = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 20
        
        if position == 0:
            # In trending regime: follow momentum
            if trending:
                if rsi_12h_aligned[i] > 60 and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
                elif rsi_12h_aligned[i] < 40 and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
            # In ranging regime: mean reversion at RSI extremes
            elif ranging:
                if rsi_12h_aligned[i] < 30 and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
                elif rsi_12h_aligned[i] > 70 and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit conditions
                if (trending and rsi_12h_aligned[i] < 50) or \
                   (ranging and rsi_12h_aligned[i] > 60) or \
                   (close[i] < close[i-1] and atr_6h[i] > 0):  # Simple trailing condition
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit conditions
                if (trending and rsi_12h_aligned[i] > 50) or \
                   (ranging and rsi_12h_aligned[i] < 40) or \
                   (close[i] > close[i-1] and atr_6h[i] > 0):  # Simple trailing condition
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals