#!/usr/bin/env python3
"""
Hypothesis: 1h volume-weighted RSI mean reversion with 4h trend filter and session timing.
- Long when: RSI(14) < 30, price > VWAP(20), and price > 4h EMA50 (uptrend filter)
- Short when: RSI(14) > 70, price < VWAP(20), and price < 4h EMA50 (downtrend filter)
- Exit: RSI returns to 40-60 range OR 4h EMA50 trend flip
- Uses 1h for precise entry timing, 4h for trend direction, volume confirmation via VWAP
- Session filter: 08-20 UTC to avoid low-liquidity periods
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
- Discrete position sizing: ±0.20 to minimize fee churn
- Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)
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
    
    # Session filter: 08-20 UTC (precompute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # RSI(14) - momentum oscillator
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # VWAP(20) - volume-weighted average price
    typical_price = (high + low + close) / 3
    vwap_num = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum()
    vwap_den = pd.Series(volume).rolling(window=20, min_periods=20).sum()
    vwap = (vwap_num / vwap_den).replace([np.inf, -np.inf], np.nan).values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # VWAP(20), RSI(14), EMA50(4h)
    
    for i in range(start_idx, n):
        # Skip if outside trading session or data not ready
        if not in_session[i] or \
           np.isnan(rsi[i]) or \
           np.isnan(vwap[i]) or \
           np.isnan(ema_50_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold, price above VWAP, and uptrend (price > 4h EMA50)
            if (rsi[i] < 30 and 
                close[i] > vwap[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought, price below VWAP, and downtrend (price < 4h EMA50)
            elif (rsi[i] > 70 and 
                  close[i] < vwap[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral range OR trend flip to downside
            if (rsi[i] > 40 and rsi[i] < 60) or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI returns to neutral range OR trend flip to upside
            if (rsi[i] > 40 and rsi[i] < 60) or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VWAP_RSI_MeanReversion_4hEMA50_Trend_Session"
timeframe = "1h"
leverage = 1.0