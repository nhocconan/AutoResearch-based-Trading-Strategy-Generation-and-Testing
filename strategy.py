#!/usr/bin/env python3
"""
12h_RVOL_Donchian20_RSI_Confirmation_v1
Hypothesis: Trade Donchian(20) breakouts on 12h with RVOL and RSI confirmation for breakout strength.
RVOL (relative volume) > 1.5 confirms institutional interest, while RSI(14) avoids overextended entries.
Works in bull/bear by capturing strong momentum moves with volume confirmation. Targets 15-30 trades/year.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # RVOL: volume / average volume of last 20 periods
    rvol_period = 20
    rvol = np.full(n, np.nan)
    if n >= rvol_period:
        vol_ma = np.convolve(volume, np.ones(rvol_period)/rvol_period, mode='same')
        # Handle edges
        for i in range(rvol_period):
            vol_ma[i] = np.mean(volume[:i+1]) if i+1 > 0 else np.nan
        for i in range(n - rvol_period + 1, n):
            vol_ma[i] = np.mean(volume[i-rvol_period+1:i+1]) if i-rvol_period+1 >= 0 else np.nan
        rvol = volume / vol_ma
        rvol[vol_ma == 0] = np.nan
    
    # RSI(14) for overextension filter
    rsi_period = 14
    rsi = np.full(n, np.nan)
    if n >= rsi_period + 1:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        for i in range(rsi_period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, rvol_period, rsi_period + 1)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(rvol[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        trend_filter_long = close[i] > ema_50_1w_aligned[i]
        trend_filter_short = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: break above upper band + RVOL > 1.5 + RSI < 70 (not overbought) + uptrend
            if (close[i] > upper[i] and rvol[i] > 1.5 and rsi[i] < 70 and trend_filter_long):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band + RVOL > 1.5 + RSI > 30 (not oversold) + downtrend
            elif (close[i] < lower[i] and rvol[i] > 1.5 and rsi[i] > 30 and trend_filter_short):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below lower band or RSI > 75 (overbought)
            if close[i] < lower[i] or rsi[i] > 75:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above upper band or RSI < 25 (oversold)
            if close[i] > upper[i] or rsi[i] < 25:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RVOL_Donchian20_RSI_Confirmation_v1"
timeframe = "12h"
leverage = 1.0