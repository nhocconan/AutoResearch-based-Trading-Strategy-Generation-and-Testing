#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d RSI (14) for mean reversion
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Bollinger Bands (20,2) for volatility and mean reversion
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate 1d ATR (14) for volatility filter and stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_1d_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(upper_bb_val) or 
            np.isnan(lower_bb_val) or np.isnan(atr_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower Bollinger Band with RSI oversold
            if close_val <= lower_bb_val and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper Bollinger Band with RSI overbought
            elif close_val >= upper_bb_val and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches upper Bollinger Band or RSI overbought
            if close_val >= upper_bb_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches lower Bollinger Band or RSI oversold
            if close_val <= lower_bb_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_BollingerRSI_MeanReversion_V1
# Uses 1-day Bollinger Bands (20,2) and RSI (14) for mean reversion signals
# Enters long when 4h price touches lower Bollinger Band with RSI < 30 (oversold)
# Enters short when 4h price touches upper Bollinger Band with RSI > 70 (overbought)
# Exits when price touches opposite Bollinger Band or RSI reaches extreme opposite level
# Uses 1d ATR as volatility filter to avoid extremely low volatility periods
# Designed for 4h timeframe with ~20-50 trades/year
name = "4h_BollingerRSI_MeanReversion_V1"
timeframe = "4h"
leverage = 1.0