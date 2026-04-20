#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly RSI with daily price action confirmation
# RSI(14) on weekly timeframe to identify overbought/oversold conditions
# Enter long when weekly RSI < 30 and daily close > daily open (bullish candle)
# Enter short when weekly RSI > 70 and daily close < daily open (bearish candle)
# Exit when RSI returns to neutral zone (40-60) or opposite signal appears
# Designed to work in both bull and bear markets by capturing mean reversion
# at extreme weekly levels while using daily price action for timing
# Target: 20-50 trades/year to minimize fee drag

name = "1d_1w_RSI_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily price action
    close = prices['close'].values
    open_price = prices['open'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        rsi_val = rsi_1w_aligned[i]
        close_val = close[i]
        open_val = open_price[i]
        
        # Skip if RSI is not ready
        if np.isnan(rsi_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly oversold + daily bullish candle
            if rsi_val < 30 and close_val > open_val:
                signals[i] = 0.25
                position = 1
            # Short: Weekly overbought + daily bearish candle
            elif rsi_val > 70 and close_val < open_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or bearish candle
            if rsi_val >= 40 or close_val < open_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or bullish candle
            if rsi_val <= 60 or close_val > open_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals