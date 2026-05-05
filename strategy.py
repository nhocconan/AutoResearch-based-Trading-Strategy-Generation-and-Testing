#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and 1d volatility regime
# Long when: 1h RSI(2) < 10 AND 4h close > 4h EMA(50) AND 1d ATR ratio < 0.8 (low volatility)
# Short when: 1h RSI(2) > 90 AND 4h close < 4h EMA(50) AND 1d ATR ratio < 0.8 (low volatility)
# Exit when: 1h RSI(2) crosses 50 (mean reversion complete) OR opposite signal occurs
# Uses RSI(2) for extreme mean reversion, 4h EMA for trend alignment, 1d ATR for regime filter
# Timeframe: 1h, HTF: 4h/1d. Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
# Session filter: 08-20 UTC to reduce noise trades.

name = "1h_RSI2_4hEMA50_1dATRRegime_VolatilityFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1h RSI(2) - extreme mean reversion
    if len(close) >= 3:
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean()
        avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_2 = 100 - (100 / (1 + rs))
        rsi_2 = rsi_2.fillna(50).values  # neutral when undefined
    else:
        rsi_2 = np.full(n, 50)
    
    # Get 4h data ONCE before loop for EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR ratio (current/20-period MA) for volatility regime
    if len(df_1d) >= 14:
        tr1 = pd.Series(df_1d['high']).values - pd.Series(df_1d['low']).values
        tr2 = np.abs(pd.Series(df_1d['high']).values - pd.Series(df_1d['close']).shift(1).values)
        tr3 = np.abs(pd.Series(df_1d['low']).values - pd.Series(df_1d['close']).shift(1).values)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
        atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
        atr_ratio = atr_14 / atr_ma_20.replace(0, np.nan)
        atr_ratio = np.nan_to_num(atr_ratio, nan=1.0)  # default to normal volatility
    else:
        atr_ratio = np.full(len(df_1d), 1.0)
    
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(rsi_2[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI(2) < 10 (extreme oversold) + 4h close > EMA(50) (uptrend) + low volatility
            if (rsi_2[i] < 10 and 
                close[i] > ema_50_4h_aligned[i] and 
                atr_ratio_aligned[i] < 0.8):
                signals[i] = 0.20
                position = 1
            # Short conditions: RSI(2) > 90 (extreme overbought) + 4h close < EMA(50) (downtrend) + low volatility
            elif (rsi_2[i] > 90 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  atr_ratio_aligned[i] < 0.8):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI(2) crosses above 50 (mean reversion complete) OR short signal
            if (rsi_2[i] > 50 or 
                (rsi_2[i] > 90 and close[i] < ema_50_4h_aligned[i] and atr_ratio_aligned[i] < 0.8)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI(2) crosses below 50 (mean reversion complete) OR long signal
            if (rsi_2[i] < 50 or 
                (rsi_2[i] < 10 and close[i] > ema_50_4h_aligned[i] and atr_ratio_aligned[i] < 0.8)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals