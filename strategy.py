#!/usr/bin/env python3
"""
6h Bollinger Bands Squeeze Breakout with Volume and ADX Trend Filter
Hypothesis: Bollinger Band squeeze indicates low volatility, often preceding breakouts.
Breakouts in direction of ADX trend (ADX>25) with volume confirmation (1.5x average) capture explosive moves.
Works in bull (buy upside breakouts) and bear (sell downside breakouts). Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14399_6h_bb_squeeze_breakout_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper = sma + (std * bb_std)
    lower = sma - (std * bb_std)
    bb_width = (upper - lower) / sma  # normalized width
    
    # Bollinger Band Squeeze: width below 20-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean()
    squeeze = bb_width < bb_width_ma
    
    # ADX (14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).sum() / (atr * period)
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).sum() / (atr * period)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean()
        return adx.fillna(0).values
    
    adx = calculate_adx(high, low, close)
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Start from warmup period
    start = max(bb_period, 20) + 14  # BB period + ADX period
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below middle Bollinger Band OR ADX weakens OR stoploss
            if (close[i] < sma[i] or adx[i] < 20 or 
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above middle Bollinger Band OR ADX weakens OR stoploss
            if (close[i] > sma[i] or adx[i] < 20 or 
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Bollinger Band breakout + ADX trend + volume
            long_setup = (close[i] > upper[i] and squeeze[i] and adx[i] > 25 and vol_filter[i])
            short_setup = (close[i] < lower[i] and squeeze[i] and adx[i] > 25 and vol_filter[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals