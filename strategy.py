#!/usr/bin/env python3
"""
6h ADX + Volume-Weighted RSI with 12h Trend Filter
Long: ADX > 25, VW-RSI < 30, price > 12h EMA50
Short: ADX > 25, VW-RSI > 70, price < 12h EMA50
Exit: ADX < 20 or price crosses 12h EMA50
Combines trend strength (ADX) with mean reversion (VW-RSI) and higher timeframe trend filter.
Designed to work in both trending and ranging markets by requiring ADX for trend and VW-RSI for entry timing.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ADX (14)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    atr = np.zeros_like(close)
    atr[0] = tr[0] if len(tr) > 0 else 0
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    plus_di = 100 * (np.convolve(plus_dm, np.ones(14)/14, mode='full')[:len(close)] / np.where(atr == 0, 1, atr))
    minus_di = 100 * (np.convolve(minus_dm, np.ones(14)/14, mode='full')[:len(close)] / np.where(atr == 0, 1, atr))
    dx = 100 * np.absolute(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = np.convolve(dx, np.ones(14)/14, mode='full')[:len(close)]
    
    # Volume-Weighted RSI (14)
    price_change = np.diff(close, prepend=close[0])
    up = np.where(price_change > 0, price_change, 0)
    down = np.where(price_change < 0, -price_change, 0)
    vw_up = up * volume
    vw_down = down * volume
    vw_rs = np.convolve(vw_up, np.ones(14)/14, mode='full')[:len(close)] / np.where(np.convolve(vw_down, np.ones(14)/14, mode='full')[:len(close)] == 0, 1, np.convolve(vw_down, np.ones(14)/14, mode='full')[:len(close)])
    vw_rsi = 100 - (100 / (1 + vw_rs))
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, 30)  # need EMA50 and ADX/VW-RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx[i]) or np.isnan(vw_rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_val = ema_50_12h_aligned[i]
        adx_val = adx[i]
        vw_rsi_val = vw_rsi[i]
        
        if position == 0:
            # Long: Strong trend + oversold + above 12h EMA50
            if adx_val > 25 and vw_rsi_val < 30 and price > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Short: Strong trend + overbought + below 12h EMA50
            elif adx_val > 25 and vw_rsi_val > 70 and price < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Weak trend or price below 12h EMA50
            if adx_val < 20 or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Weak trend or price above 12h EMA50
            if adx_val < 20 or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_VWRSI_12hEMA50"
timeframe = "6h"
leverage = 1.0