#!/usr/bin/env python3
"""
1d_1w_CCI_TopBottom_Reversal
Hypothesis: Weekly CCI extremes combined with daily mean reversion provide high-probability
reversal signals in both bull and bear markets. Uses weekly CCI(20) > 100 for overbought
and < -100 for oversold, with daily RSI(14) for entry timing and volume confirmation.
Targets 8-15 trades/year to minimize fee drag while capturing major reversals.
"""

name = "1d_1w_CCI_TopBottom_Reversal"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume spike: >1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Weekly data for CCI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly CCI(20): (Typical Price - SMA) / (0.015 * Mean Deviation)
    tp_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    sma_20 = tp_1w.rolling(window=20, min_periods=20).mean()
    mad = tp_1w.rolling(window=20, min_periods=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci_20 = (tp_1w - sma_20) / (0.015 * mad)
    cci_20 = cci_20.fillna(0).values
    
    # Align weekly CCI to daily timeframe
    cci_20_aligned = align_ltf_to_htf(prices, df_1w, cci_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if np.isnan(cci_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Weekly oversold + daily RSI < 30 + volume spike
            if (cci_20_aligned[i] < -100 and 
                rsi[i] < 30 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly overbought + daily RSI > 70 + volume spike
            elif (cci_20_aligned[i] > 100 and 
                  rsi[i] > 70 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly CCI crosses above -50 OR daily RSI > 70
            if (cci_20_aligned[i] > -50 or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly CCI crosses below 50 OR daily RSI < 30
            if (cci_20_aligned[i] < 50 or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals