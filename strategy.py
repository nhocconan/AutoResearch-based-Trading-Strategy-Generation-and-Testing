#!/usr/bin/env python3
"""
4h_1D_KAMA_Trend_RSI_Momentum
Hypothesis: 4h timeframe using 1d KAMA trend direction and RSI momentum for entries.
KAMA adapts to market noise, reducing whipsaw in choppy markets. RSI filters for momentum strength.
Works in bull markets (trend + momentum) and bear markets (only takes counter-trend bounces when RSI extreme).
Target: 20-40 trades/year by requiring both trend alignment and RSI extreme.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1D_KAMA_Trend_RSI_Momentum"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility as sum of absolute changes over ER period
    er_period = 10
    change_vec = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_sum = np.convolve(change_vec, np.ones(er_period), 'same')
    volatility_sum[:er_period-1] = np.cumsum(change_vec[:er_period-1])[::-1][:er_period-1]  # fix edges
    er = np.where(volatility_sum != 0, change_vec / volatility_sum, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # Calculate 1d RSI (14 period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align to 4h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume average (20 period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > average
        volume_confirm = volume[i] > vol_ma[i]
        
        # Trend and momentum conditions
        above_kama = close[i] > kama_1d_aligned[i]
        below_kama = close[i] < kama_1d_aligned[i]
        rsi_overbought = rsi_1d_aligned[i] > 70
        rsi_oversold = rsi_1d_aligned[i] < 30
        
        # Entry conditions:
        # Long: price above KAMA (uptrend) AND RSI not overbought AND volume confirmation
        # Short: price below KAMA (downtrend) AND RSI not oversold AND volume confirmation
        long_entry = above_kama and (not rsi_overbought) and volume_confirm
        short_entry = below_kama and (not rsi_oversold) and volume_confirm
        
        # Exit conditions:
        # Long exit: price crosses below KAMA OR RSI becomes overbought
        # Short exit: price crosses above KAMA OR RSI becomes oversold
        long_exit = (not above_kama) or rsi_overbought
        short_exit = (not below_kama) or rsi_oversold
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals