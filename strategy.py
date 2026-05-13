#!/usr/bin/env python3
"""
4h_1D_RSI_Backtest
Strategy: Use 4h timeframe with 1D RSI filter and volume confirmation.
Enters long when price is above 4h EMA50, 1D RSI < 30 (oversold), and volume spike.
Enters short when price is below 4h EMA50, 1D RSI > 70 (overbought), and volume spike.
Exits when RSI returns to neutral (40-60) or trend reversal.
Position size 0.25 to limit risk and trade frequency.
Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
"""

name = "4h_1D_RSI_Backtest"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1D RSI to 4h chart (wait for daily close)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 4h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price above EMA50, 1D RSI oversold (<30), volume spike
            if (close[i] > ema50[i] and 
                rsi_1d_aligned[i] < 30 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below EMA50, 1D RSI overbought (>70), volume spike
            elif (close[i] < ema50[i] and 
                  rsi_1d_aligned[i] > 70 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>=40) or trend reversal
            if (rsi_1d_aligned[i] >= 40) or (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<=60) or trend reversal
            if (rsi_1d_aligned[i] <= 60) or (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals