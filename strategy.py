#!/usr/bin/env python3
"""
Hypothesis: 6h Volume-Weighted RSI with 12h EMA Trend Filter and ATR-Based Exit.
Long when VWRSI < 30 (oversold) AND price > 12h EMA34 (uptrend).
Short when VWRSI > 70 (overbought) AND price < 12h EMA34 (downtrend).
Exit when VWRSI crosses 50 (mean reversion) OR ATR(14) expansion signals exhaustion.
Uses 12h for EMA34 trend filter, 6h for VWRSI calculation.
Target: 50-150 total trades over 4 years (12-37/year). VWRSI improves on classic RSI by weighting price by volume,
reducing false signals in low-volume chop. ATR-based exit adapts to volatility, avoiding premature exits in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Volume-Weighted RSI on 6h timeframe (period=14)
    delta = pd.Series(close).diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    
    # Volume-weighted gains and losses
    vol_up = (up * pd.Series(volume)).ewm(span=14, adjust=False, min_periods=14).mean()
    vol_down = (down * pd.Series(volume)).ewm(span=14, adjust=False, min_periods=14).mean()
    
    rs = vol_up / vol_down.replace(0, np.nan)
    vwrsi = 100 - (100 / (1 + rs))
    vwrsi = vwrsi.fillna(50).values  # neutral when no volume
    
    # Align 12h EMA34 to 6h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate ATR(14) for volatility-based exit
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(vwrsi[i]):
            signals[i] = 0.0
            continue
        
        rsi = vwrsi[i]
        price = close[i]
        ema34 = ema34_12h_aligned[i]
        volatility = atr[i]
        
        if position == 0:
            # Long: VWRSI < 30 (oversold) AND price > 12h EMA34 (uptrend)
            if rsi < 30 and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: VWRSI > 70 (overbought) AND price < 12h EMA34 (downtrend)
            elif rsi > 70 and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VWRSI crosses above 50 (mean reversion) OR ATR expansion > 1.5x (exhaustion)
            if rsi > 50 or (i > start_idx and atr[i] > 1.5 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VWRSI crosses below 50 (mean reversion) OR ATR expansion > 1.5x (exhaustion)
            if rsi < 50 or (i > start_idx and atr[i] > 1.5 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolumeWeightedRSI_12hEMA34_ATRExit"
timeframe = "6h"
leverage = 1.0