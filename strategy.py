#!/usr/bin/env python3
"""
1d_rsi_ema_trend_filter_v1
Hypothesis: On daily timeframe, use EMA(50) trend filter combined with RSI(14) mean reversion entries.
In bull markets, buy dips below RSI 40 in uptrend (price > EMA50). In bear markets, sell rallies above RSI 60 in downtrend (price < EMA50).
Weekly RSI(14) acts as regime filter: only take longs when weekly RSI > 50, shorts when weekly RSI < 50.
This reduces whipsaws and focuses on higher probability trades. Target: 15-25 trades/year (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_rsi_ema_trend_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA(50) for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14) on daily
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly RSI(14) as regime filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    delta_w = np.diff(close_weekly, prepend=close_weekly[0])
    gain_w = np.where(delta_w > 0, delta_w, 0)
    loss_w = np.where(delta_w < 0, -delta_w, 0)
    gain_ma_w = pd.Series(gain_w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma_w = pd.Series(loss_w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_w = gain_ma_w / (loss_ma_w + 1e-10)
    rsi_weekly = 100 - (100 / (1 + rs_w))
    rsi_weekly_aligned = align_htf_to_ltf(prices, df_weekly, rsi_weekly)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema50[i]) or np.isnan(rsi[i]) or np.isnan(rsi_weekly_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Weekly regime filter
        weekly_bull = rsi_weekly_aligned[i] > 50
        weekly_bear = rsi_weekly_aligned[i] < 50
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend change (price < EMA50)
            if rsi[i] > 70 or close[i] < ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend change (price > EMA50)
            if rsi[i] < 30 or close[i] > ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above EMA50 (uptrend) + RSI < 40 (oversold) + weekly bullish
            if close[i] > ema50[i] and rsi[i] < 40 and weekly_bull:
                position = 1
                signals[i] = 0.25
            # Short: price below EMA50 (downtrend) + RSI > 60 (overbought) + weekly bearish
            elif close[i] < ema50[i] and rsi[i] > 60 and weekly_bear:
                position = -1
                signals[i] = -0.25
    
    return signals