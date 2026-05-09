#!/usr/bin/env python3
name = "6H_Contrarian_Kelly_Volume_Regime_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Kelly criterion and regime detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily returns for Kelly criterion
    daily_returns = np.diff(close_1d) / close_1d[:-1]
    daily_returns = np.concatenate([[0], daily_returns])  # align with close_1d
    
    # Calculate win rate and win/loss ratio over 60-day window
    win_rate = np.zeros_like(close_1d)
    win_loss_ratio = np.ones_like(close_1d)  # default to 1 to avoid division by zero
    
    for i in range(60, len(close_1d)):
        window_returns = daily_returns[i-60:i]
        wins = window_returns[window_returns > 0]
        losses = window_returns[window_returns < 0]
        
        if len(wins) > 0 and len(losses) > 0:
            win_rate[i] = len(wins) / len(window_returns)
            avg_win = np.mean(wins)
            avg_loss = np.mean(np.abs(losses))
            win_loss_ratio[i] = avg_win / avg_loss if avg_loss > 0 else 1
        else:
            win_rate[i] = 0.5  # neutral
            win_loss_ratio[i] = 1
    
    # Kelly fraction: f = (bp - q) / b where b = win_loss_ratio, p = win_rate, q = 1-p
    kelly_fraction = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        p = win_rate[i]
        q = 1 - p
        b = win_loss_ratio[i]
        kelly_fraction[i] = max(0, (b * p - q) / b) if b > 0 else 0
        # Cap Kelly at 0.5 to avoid over-leverage
        kelly_fraction[i] = min(kelly_fraction[i], 0.5)
    
    # Align Kelly fraction to 6h timeframe
    kelly_aligned = align_htf_to_ltf(prices, df_1d, kelly_fraction)
    
    # Regime detection: volatility regime using ATR ratio
    # Calculate daily ATR(10) and ATR(30)
    atr10_1d = np.zeros_like(close_1d)
    atr30_1d = np.zeros_like(close_1d)
    
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[0], tr1])
    
    tr30 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr30 = np.maximum(tr30, np.abs(low_1d[1:] - close_1d[:-1]))
    tr30 = np.concatenate([[0], tr30])
    
    for i in range(1, len(close_1d)):
        if i >= 10:
            atr10_1d[i] = np.mean(tr1[max(0, i-9):i+1])
        if i >= 30:
            atr30_1d[i] = np.mean(tr30[max(0, i-29):i+1])
    
    # Avoid division by zero
    atr30_1d[atr30_1d == 0] = 1e-10
    atr_ratio = atr10_1d / atr30_1d  # high = volatile, low = calm
    
    # Align ATR ratio to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: 6h volume vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after volume MA warmup
        # Skip if data not ready
        if np.isnan(kelly_aligned[i]) or np.isnan(atr_ratio_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in low volatility (calm markets) for mean reversion
        # ATR ratio < 0.6 indicates low volatility regime
        low_vol_regime = atr_ratio_aligned[i] < 0.6
        
        # Mean reversion signal: extreme Kelly + low volatility
        # High Kelly suggests edge exists, but we fade extreme readings in calm markets
        kelly_extreme = kelly_aligned[i] > 0.3  # strong edge detected
        
        # Price deviation from 20-period mean for entry timing
        price_deviation = (close[i] - np.mean(close[max(0, i-19):i+1])) / np.std(close[max(0, i-19):i+1]) if i >= 19 and np.std(close[max(0, i-19):i+1]) > 0 else 0
        
        if position == 0:
            # Enter long: extreme negative deviation + Kelly edge + low vol
            if price_deviation < -1.5 and kelly_extreme and low_vol_regime and volume[i] > vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: extreme positive deviation + Kelly edge + low vol
            elif price_deviation > 1.5 and kelly_extreme and low_vol_regime and volume[i] > vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reverts to mean or volatility increases
            if price_deviation > -0.5 or atr_ratio_aligned[i] > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reverts to mean or volatility increases
            if price_deviation < 0.5 or atr_ratio_aligned[i] > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals