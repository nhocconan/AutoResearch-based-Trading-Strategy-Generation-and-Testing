#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA34 + 1d RSI mean reversion with volume confirmation
# - Uses 6h EMA34 for trend direction: long when close > EMA34, short when close < EMA34
# - Entry: RSI(14) on 1d < 35 for long or > 65 for short, with volume > 1.5x 20-period average
# - Exit: price crosses back over EMA34
# - Volume confirmation reduces false signals
# - Target: 20-35 trades per year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 6h data for EMA calculation
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    
    # Calculate EMA34 on 6h data
    ema_34 = pd.Series(close_6h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_6h, ema_34)
    
    # Load 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1d data
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_6h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6h price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema_34_6h[i]) or np.isnan(rsi_1d_6h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price above EMA34 + RSI oversold + volume surge
            if price > ema_34_6h[i] and rsi_1d_6h[i] < 35 and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price below EMA34 + RSI overbought + volume surge
            elif price < ema_34_6h[i] and rsi_1d_6h[i] > 65 and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below EMA34
            if price < ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA34
            if price > ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA34_1dRSI_MeanReversion_Volume"
timeframe = "6h"
leverage = 1.0