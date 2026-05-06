#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Relative Strength Index (RSI) with volume confirmation and volatility filter
# Long when RSI(14) < 30 (oversold) and volume > 1.5x 20-period average
# Short when RSI(14) > 70 (overbought) and volume > 1.5x 20-period average
# Uses daily RSI for overbought/oversold conditions, volume for confirmation, and volatility filter to avoid choppy markets
# Designed to work in bull markets via mean reversion from oversold levels and in bear markets via mean reversion from overbought levels
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "4h_1dRSI14_Volume_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day RSI (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI using standard formula
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align RSI to 4h timeframe
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Volatility filter: avoid choppy markets (ATR ratio < 0.5 indicates low volatility/chop)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (0.5 * atr_ma_50)  # Only trade when volatility is above 50% of its 50-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after RSI and ATR warmup
        # Skip if any critical value is NaN
        if (np.isnan(rsi_4h[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: RSI oversold (<30) with volume confirmation and sufficient volatility
            if rsi_4h[i] < 30 and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (>70) with volume confirmation and sufficient volatility
            elif rsi_4h[i] > 70 and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (>50) or overbought (>70)
            if rsi_4h[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral (<50) or oversold (<30)
            if rsi_4h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals