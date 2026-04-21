#!/usr/bin/env python3
"""
6h_RSI_Bollinger_Band_Reversal
Hypothesis: Combines RSI overbought/oversold conditions with Bollinger Band mean reversion on the 6h timeframe, filtered by 1d trend (EMA50). In bull markets, buy oversold dips in uptrend; in bear markets, sell overbought rallies in downtrend. Bollinger Bands provide dynamic support/resistance, while RSI identifies exhaustion. The 1d EMA50 filter ensures we trade with the higher timeframe trend, reducing false signals during sideways periods. Designed for low trade frequency (target: 15-30/year) to minimize fee drag in 6h timeframe. Uses discrete position sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.zeros_like(close_1d)
    ema50_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6s RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(gain)):
        if i < 14:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else gain[i]
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 6s Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = np.zeros_like(close)
    bb_std_dev = np.zeros_like(close)
    for i in range(n):
        if i < bb_period:
            sma[i] = np.mean(close[:i+1])
            bb_std_dev[i] = np.std(close[:i+1]) if i > 0 else 0.0
        else:
            sma[i] = np.mean(close[i-bb_period+1:i+1])
            bb_std_dev[i] = np.std(close[i-bb_period+1:i+1])
    upper_band = sma + (bb_std_dev * bb_std)
    lower_band = sma - (bb_std_dev * bb_std)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema50_1d_aligned[i]
        rsi_val = rsi[i]
        upper = upper_band[i]
        lower = lower_band[i]
        
        # Exit conditions
        if position == 1:
            # Exit long: price touches upper band or RSI overbought
            if price >= upper or rsi_val >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches lower band or RSI oversold
            if price <= lower or rsi_val <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        # Entry conditions
        if position == 0:
            # Long: RSI oversold (<30), price near/below lower band, and 1d uptrend (price > EMA50)
            if rsi_val < 30 and price <= lower and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: RSI overbought (>70), price near/above upper band, and 1d downtrend (price < EMA50)
            elif rsi_val > 70 and price >= upper and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
    
    return signals

name = "6h_RSI_Bollinger_Band_Reversal"
timeframe = "6h"
leverage = 1.0