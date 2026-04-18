#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

"""
Hypothesis: 1h momentum with 4h trend filter and volume confirmation.
In bull markets: 4h EMA20 up + 1h price > EMA13 + volume spike = long.
In bear markets: 4h EMA20 down + 1h price < EMA13 + volume spike = short.
Uses 1h for entry timing, 4h for direction to avoid overtrading.
Target: 20-40 trades/year per symbol.
"""

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA20 for trend
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1h EMA13 for entry
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume spike (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need for EMA20
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_13[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend
        uptrend_4h = ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1] if i > 0 else False
        downtrend_4h = ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1] if i > 0 else False
        
        # 1h price vs EMA
        price_above_ema = close[i] > ema_13[i]
        price_below_ema = close[i] < ema_13[i]
        
        if position == 0:
            # Long: 4h uptrend + price > EMA13 + volume spike
            if uptrend_4h and price_above_ema and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + price < EMA13 + volume spike
            elif downtrend_4h and price_below_ema and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h trend change or price < EMA13
            if not uptrend_4h or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h trend change or price > EMA13
            if not downtrend_4h or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA13_EMA20_Trend_Volume"
timeframe = "1h"
leverage = 1.0