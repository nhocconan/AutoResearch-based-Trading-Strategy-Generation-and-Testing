#!/usr/bin/env python3
"""
1h RSI(2) Pullback + 4h Trend + Volume Confirmation
Hypothesis: In strong 4h trends, buy pullbacks to RSI(2) oversold in uptrends and sell rallies to RSI(2) overbought in downtrends.
Uses 4h EMA(50) for trend filter and RSI(2) on 1h for precise entry timing. Volume filter avoids low-activity periods.
Designed for 60-120 trades over 4 years (15-30/year) with 0.20 position sizing to manage drawdowns.
Works in bull (buy uptrend pullbacks) and bear (sell downtrend rallies) by following 4h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14354_1h_rsi2_pullback_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend direction
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI(2) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
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
    start = max(50, 2, 20, 14)  # EMA50, RSI2, volMA20, ATR14
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Determine 4h trend
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI(2) overbought OR trend reversal OR stoploss
            if (rsi[i] >= 80 or not uptrend or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI(2) oversold OR trend reversal OR stoploss
            if (rsi[i] <= 20 or not downtrend or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: pullback to RSI(2) extreme in direction of 4h trend
            long_setup = (rsi[i] <= 15 and uptrend and vol_filter[i])
            short_setup = (rsi[i] >= 85 and downtrend and vol_filter[i])
            
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