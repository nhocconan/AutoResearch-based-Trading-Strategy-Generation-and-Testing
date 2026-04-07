#!/usr/bin/env python3
"""
4h ADX Trend Strength + Bollinger Squeeze Breakout
Long when ADX > 25 (trending) and price breaks above upper BB after low volatility (BBW < 50th percentile)
Short when ADX > 25 and price breaks below lower BB after low volatility
Exit when ADX < 20 (trend weakening) or opposite breakout occurs
Designed to capture strong trends after consolidation periods in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_adx_bb_squeeze_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === ADX (14) ===
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === Bollinger Bands (20, 2) ===
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma + 2 * std
    lower_bb = sma - 2 * std
    bb_width = (upper_bb - lower_bb) / (sma + 1e-10)
    
    # Bollinger width percentile (20-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # === 1d ADX Trend Filter (Higher Timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(adx[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(bb_width_percentile[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend weakening or opposite breakout
            if adx[i] < 20 or close[i] < lower_bb[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakening or opposite breakout
            if adx[i] < 20 or close[i] > upper_bb[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout after low volatility (squeeze) with trend confirmation
            if (adx[i] > 25 and adx_1d_aligned[i] > 20 and 
                bb_width_percentile[i] < 0.5):  # Low volatility environment
                if close[i] > upper_bb[i]:  # Break above upper band
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lower_bb[i]:  # Break below lower band
                    position = -1
                    signals[i] = -0.25
    
    return signals