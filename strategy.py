#!/usr/bin/env python3
# 4h_1d_rsi_volume_momentum_v1
# Strategy: 4h RSI momentum with 1d volume confirmation and trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: RSI momentum (RSI > 60 for long, RSI < 40 for short) combined with
# above-average 1d volume and aligned with 1d EMA50 trend captures momentum bursts.
# Uses tight entry conditions to limit trades (<30/year) and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_rsi_volume_momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: avoid low volatility (range) markets
        # ATR ratio: current ATR vs 50-period average ATR
        if i >= 50:
            atr_ma = pd.Series(atr[:i+1]).rolling(window=50, min_periods=50).mean().iloc[-1]
            vol_filter = atr[i] > 0.8 * atr_ma  # Only trade when volatility is above 80% of average
        else:
            vol_filter = True
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: close vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # RSI momentum: RSI > 60 for long, RSI < 40 for short
        rsi_long = rsi[i] > 60
        rsi_short = rsi[i] < 40
        
        # Entry conditions
        # Long: RSI > 60 AND uptrend AND volume confirmation AND volatility filter
        if rsi_long and uptrend and vol_confirm and vol_filter and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: RSI < 40 AND downtrend AND volume confirmation AND volatility filter
        elif rsi_short and downtrend and vol_confirm and vol_filter and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: RSI returns to neutral zone (40-60) - mean reversion signal
        elif position == 1 and rsi[i] < 50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi[i] > 50:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals