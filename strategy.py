#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI4_Div_Liquidity"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI(4) for mean reversion signals
    close_d = get_htf_data(prices, '1d')
    if len(close_d) < 10:
        return np.zeros(n)
    
    close_series = pd.Series(close_d['close'].values)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = loss.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi4 = 100 - (100 / (1 + rs))
    rsi4_values = rsi4.fillna(50).values
    rsi4_aligned = align_htf_to_ltf(prices, close_d, rsi4_values)
    
    # Liquidity zones: intraday high/low of current 6h bar vs previous bar
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(rsi4_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price breaks above prior 6h high + volume
            if rsi4_aligned[i] < 30 and close[i] > high_prev[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + price breaks below prior 6h low + volume
            elif rsi4_aligned[i] > 70 and close[i] < low_prev[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion) or stop below prior low
            if rsi4_aligned[i] > 50 or close[i] < low_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 or stop above prior high
            if rsi4_aligned[i] < 50 or close[i] > high_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals