#!/usr/bin/env python3
"""
1d RSI + Volume Spike + CCI Trend Filter
Hypothesis: RSI identifies momentum extremes while volume confirms institutional participation.
Long when RSI(14) crosses above 30 with volume spike and CCI(20) > 0.
Short when RSI(14) crosses below 70 with volume spike and CCI(20) < 0.
Exit on RSI reversal or 2*ATR stoploss. Works in bull (buy dips) and bear (sell rallies).
Target: 60-100 total trades over 4 years (15-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14324_1d_rsi_vol_cci_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate 20-period EMA for weekly trend filter
    ema_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # CCI (20)
    tp = (high + low + close) / 3
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    md = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - ma_tp) / (0.015 * md + 1e-10)
    
    # Volume filter: spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)  # Require 150% of average volume
    
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
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(cci[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(ema_weekly_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI drops below 50 OR stoploss
            if rsi[i] < 50 or close[i] <= entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI rises above 50 OR stoploss
            if rsi[i] > 50 or close[i] >= entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: RSI extreme + volume spike + weekly trend filter
            long_setup = (rsi[i-1] <= 30) and (rsi[i] > 30) and vol_spike[i] and (cci[i] > 0) and (close[i] > ema_weekly_aligned[i])
            short_setup = (rsi[i-1] >= 70) and (rsi[i] < 70) and vol_spike[i] and (cci[i] < 0) and (close[i] < ema_weekly_aligned[i])
            
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