#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d RSI(14) for mean reversion
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = np.append(np.full(13, np.nan), rsi[13:])
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1w EMA(50) for long-term trend
    close_1w = df_1w['close'].values
    ema_50w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        ema_50w_val = ema_50w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(atr_1d_val) or 
            np.isnan(ema_50w_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold in up trend (weekly EMA50 rising)
            if rsi_val < 30 and ema_50w_val > ema_50w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in down trend (weekly EMA50 falling)
            elif rsi_val > 70 and ema_50w_val < ema_50w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or price below weekly EMA50
            if rsi_val > 70 or close_val < ema_50w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or price above weekly EMA50
            if rsi_val < 30 or close_val > ema_50w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_RSI_EMA50Trend_MeanReversion_V1
# Uses 1-week EMA50 for long-term trend direction
# Enters long when daily RSI < 30 (oversold) and weekly trend is up
# Enters short when daily RSI > 70 (overbought) and weekly trend is down
# Exits when RSI reverses or price crosses weekly EMA50
# Designed for 1d timeframe with ~7-25 trades/year
name = "1d_RSI_EMA50Trend_MeanReversion_V1"
timeframe = "1d"
leverage = 1.0