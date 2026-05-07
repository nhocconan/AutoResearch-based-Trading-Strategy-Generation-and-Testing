#!/usr/bin/env python3
name = "1h_4h_1d_Trend_Pullback_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend and pullback
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(21) for trend direction
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 4h EMA(50) for trend strength filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Daily EMA(50) for higher timeframe trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h RSI(14) for pullback entry
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_vals = rsi.values
    
    # 1h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: 4h uptrend + price pullback to 4h EMA21 + RSI oversold + daily uptrend
            uptrend_4h = ema_21_4h_aligned[i] > ema_50_4h_aligned[i]
            uptrend_1d = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            pullback = close[i] <= ema_21_4h_aligned[i] + 0.5 * atr[i]
            rsi_oversold = rsi_vals[i] < 40
            
            if uptrend_4h and uptrend_1d and pullback and rsi_oversold and in_session:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + price pullback to 4h EMA21 + RSI overbought + daily downtrend
            elif (not uptrend_4h) and (not uptrend_1d) and (close[i] >= ema_21_4h_aligned[i] - 0.5 * atr[i]) and (rsi_vals[i] > 60) and in_session:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: 4h trend reversal or RSI overbought
            if (ema_21_4h_aligned[i] <= ema_50_4h_aligned[i]) or (rsi_vals[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: 4h trend reversal or RSI oversold
            if (ema_21_4h_aligned[i] >= ema_50_4h_aligned[i]) or (rsi_vals[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h trend pullback strategy using 4h trend direction and daily trend filter
# - Uses 4h EMA21/EMA50 crossover for trend direction (avoids whipsaws)
# - Enters on 1h pullbacks to 4h EMA21 during trend (buys dips in uptrend, sells rallies in downtrend)
# - RSI(14) for entry timing (oversold in uptrend, overbought in downtrend)
# - Daily EMA50 ensures alignment with higher timeframe trend
# - ATR-based pullback zone adapts to volatility
# - Session filter (08-20 UTC) reduces noise from low-volume periods
# - Position size 0.20 manages risk during drawdowns
# - Target: 15-30 trades/year to avoid fee drag
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
# - Simple, robust logic with minimal overfitting risk
# - Avoids saturated indicator combinations (no pivots, no Donchian, no CMF)