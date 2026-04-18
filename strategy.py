#!/usr/bin/env python3
"""
4h_Trend_Follow_Volume_Confirm
Strategy: 4h trend following with volume confirmation and ATR filter.
Long when price > EMA50 with volume confirmation and ATR volatility filter.
Short when price < EMA50 with volume confirmation and ATR volatility filter.
Designed for 4h timeframe: ~20-30 trades/year per symbol (80-120 total over 4 years).
Uses 1d EMA200 for trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate EMA50 on 4h
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_50[i]) or np.isnan(ema_200_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA50 vs EMA200 (daily)
        uptrend = ema_50[i] > ema_200_aligned[i]
        downtrend = ema_50[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: uptrend + volume + volatility filter
            if uptrend and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + volatility filter
            elif downtrend and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or volatility collapse
            if not uptrend or not vol_filter:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or volatility collapse
            if not downtrend or not vol_filter:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Trend_Follow_Volume_Confirm"
timeframe = "4h"
leverage = 1.0