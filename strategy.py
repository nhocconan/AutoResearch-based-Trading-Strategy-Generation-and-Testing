#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h volume-weighted VWAP deviation with 12h trend filter
# - Long when price > VWAP + 0.5*ATR and 12h EMA34 > 12h EMA89 (uptrend)
# - Short when price < VWAP - 0.5*ATR and 12h EMA34 < 12h EMA89 (downtrend)
# - Exit when price crosses VWAP or ATR stop hit (1.5*ATR)
# - Uses VWAP for mean reversion, EMA crossover for trend filter, ATR for risk
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 and EMA89 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_12h = pd.Series(close_12h).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema89_12h)
    
    # Calculate ATR for stop loss and VWAP bands (using 4h data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate VWAP (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_den = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(ema89_12h_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vwap[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long entry: price above VWAP + 0.5*ATR and 12h EMA34 > EMA89 (uptrend)
            if price > vwap[i] + 0.5 * atr[i] and ema34_12h_aligned[i] > ema89_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price below VWAP - 0.5*ATR and 12h EMA34 < EMA89 (downtrend)
            elif price < vwap[i] - 0.5 * atr[i] and ema34_12h_aligned[i] < ema89_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below VWAP OR ATR stop hit (1.5*ATR)
            if price < vwap[i] or price < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above VWAP OR ATR stop hit (1.5*ATR)
            if price > vwap[i] or price > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VWAP_TrendFilter_ATRStop"
timeframe = "4h"
leverage = 1.0