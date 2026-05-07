#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h trend (EMA21) and 1d volatility regime (ATR ratio) for direction,
# with 1h RSI for entry timing. Long in uptrend + low volatility on RSI pullback,
# Short in downtrend + low volatility on RSI bounce. Designed for 15-30 trades/year.
# Works in bull/bear by following 4h trend and avoiding high volatility whipsaws.
name = "1h_4hTrend_1dVolRegime_RSI"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 4h data for trend (EMA21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA21 for trend direction
    ema_21_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Load 1d data for volatility regime (ATR ratio)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ATR(14) and ATR(50) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr = []
    for i in range(len(close_1d)):
        if i == 0:
            tr.append(high_1d[0] - low_1d[0])
        else:
            tr.append(max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1])))
    tr = np.array(tr)
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio: short-term / long-term volatility
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 1.0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 1h RSI(14) for entry timing
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend and volatility regime
        uptrend = close[i] > ema_21_4h_aligned[i]
        downtrend = close[i] < ema_21_4h_aligned[i]
        low_volatility = atr_ratio_aligned[i] < 0.8  # Volatility contraction
        
        if position == 0:
            # Long: uptrend + low volatility + RSI oversold (<30)
            if uptrend and low_volatility and rsi[i] < 30:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + low volatility + RSI overbought (>70)
            elif downtrend and low_volatility and rsi[i] > 70:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: trend reversal or volatility expansion or RSI overbought
            if (not uptrend) or (atr_ratio_aligned[i] > 1.2) or (rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: trend reversal or volatility expansion or RSI oversold
            if (not downtrend) or (atr_ratio_aligned[i] > 1.2) or (rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals