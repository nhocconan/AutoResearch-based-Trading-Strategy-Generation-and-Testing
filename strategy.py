#!/usr/bin/env python3
"""
1h RSI(2) Pullback + 4h Trend + Volume Filter
Hypothesis: In trending markets (4h EMA50 direction), RSI(2) pullbacks offer high-probability entries.
Long when 4h trend up, RSI(2) < 10, and price > EMA20 (1h). Short when 4h trend down, RSI(2) > 90, and price < EMA20.
Uses volume confirmation to avoid low-volatility traps. Works in bull (buy pullbacks in uptrend) and 
bear (sell bounces in downtrend). Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14374_1h_rsi2_pullback_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA50 for trend direction
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(2) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi2 = 100 - (100 / (1 + rs))
    rsi2 = rsi2.fillna(50).values  # neutral when undefined
    
    # EMA20 on 1h for dynamic support/resistance
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.7 * vol_ma)  # Require at least 70% of average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # EMA50 needs 50 periods
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi2[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Determine 4h trend
        trend_up = ema50_4h_aligned[i] > ema50_4h_aligned[i-1] if i > 0 else ema50_4h_aligned[i] > close_4h[0]
        trend_down = ema50_4h_aligned[i] < ema50_4h_aligned[i-1] if i > 0 else ema50_4h_aligned[i] < close_4h[0]
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI(2) > 70 (overbought) OR price < EMA20 OR stoploss
            if (rsi2[i] > 70 or close[i] < ema20[i] or 
                close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI(2) < 30 (oversold) OR price > EMA20 OR stoploss
            if (rsi2[i] < 30 or close[i] > ema20[i] or 
                close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI(2) extreme + 4h trend + EMA20 filter + volume
            long_setup = (rsi2[i] < 10 and trend_up and close[i] > ema20[i] and vol_filter[i])
            short_setup = (rsi2[i] > 90 and trend_down and close[i] < ema20[i] and vol_filter[i])
            
            if long_setup:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals