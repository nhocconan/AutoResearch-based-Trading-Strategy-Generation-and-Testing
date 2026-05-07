#!/usr/bin/env python3
name = "4h_4h_Trend_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1d data ONCE for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA20 for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d ADX for regime filter (trending vs ranging)
    plus_dm = np.where((df_1d['high'].values[1:] - df_1d['high'].values[:-1]) > (df_1d['low'].values[:-1] - df_1d['low'].values[1:]), 
                       np.maximum(df_1d['high'].values[1:] - df_1d['high'].values[:-1], 0), 0)
    minus_dm = np.where((df_1d['low'].values[:-1] - df_1d['low'].values[1:]) > (df_1d['high'].values[1:] - df_1d['high'].values[:-1]), 
                        np.maximum(df_1d['low'].values[:-1] - df_1d['low'].values[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.maximum(
        np.maximum(df_1d['high'].values - df_1d['low'].values, np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))),
        np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    )
    tr[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_20[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Mean reversion conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: RSI oversold in uptrend (ADX > 25) or ranging market
            if rsi_oversold and vol_condition and (adx_aligned[i] > 25 and close[i] > ema_20[i] or adx_aligned[i] <= 25):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in downtrend (ADX > 25) or ranging market
            elif rsi_overbought and vol_condition and (adx_aligned[i] > 25 and close[i] < ema_20[i] or adx_aligned[i] <= 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral or reversal signal
            if rsi[i] > 50 or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral or reversal signal
            if rsi[i] < 50 or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h RSI mean reversion with ADX regime filter and volume confirmation
# - In ranging markets (ADX <= 25): trade RSI extremes ( <30 long, >70 short) with volume spike
# - In trending markets (ADX > 25): trade RSI extremes only in direction of trend (4h EMA20)
# - Volume confirmation (1.5x average) reduces false signals
# - Exit when RSI returns to neutral levels (50 for long, 40 for short) to avoid overstay
# - Works in both bull and bear markets via regime adaptation
# - Position size 0.25 limits drawdown during trends
# - Target: 20-50 trades/year to avoid fee drag while capturing mean reversion edges
# - Combines proven mean reversion (RSI) with regime filtering (ADX) for robustness
# - Avoids overtrading by requiring volume confirmation and regime alignment