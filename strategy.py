#!/usr/bin/env python3
"""
4h_RSI_Regime_Reversal_34_EMA_Top_Bottom
Hypothesis: RSI extremes (oversold/overbought) combined with EMA34 trend filter and regime detection (low volatility) work in both bull and bear markets. The EMA filter avoids counter-trend trades, regime filter prevents whipsaw in chop, and tight RSI thresholds limit trades to avoid fee drag. Target: 20-40 trades/year per symbol.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ATR(14) for volatility regime
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0], low[0], abs(high[0]-close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volatility regime: low volatility when ATR < 20-period average
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    low_vol = atr < (atr_ma_20 * 0.8)  # Low volatility regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(low_vol[i])):
            signals[i] = 0.0
            continue
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Trend filter from daily EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions: RSI extreme + trend alignment + low volatility
        long_entry = rsi_oversold and uptrend and low_vol[i]
        short_entry = rsi_overbought and downtrend and low_vol[i]
        
        # Exit when RSI returns to neutral zone (40-60)
        long_exit = rsi[i] > 40
        short_exit = rsi[i] < 60
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0  # Exit to flat
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0  # Exit to flat
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_Regime_Reversal_34_EMA_Top_Bottom"
timeframe = "4h"
leverage = 1.0