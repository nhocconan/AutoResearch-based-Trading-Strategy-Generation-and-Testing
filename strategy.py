#!/usr/bin/env python3
"""
4h 12h EMA Trend + Volume Spike + RSI Filter
Hypothesis: Trend alignment on 12h (EMA34) filters false breakouts, volume spikes confirm institutional interest,
and RSI avoids overextended entries. This combination reduces whipsaw in both bull and bear markets.
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
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume spike detection: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # RSI(14) to avoid overextended entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        trend = ema34_12h_aligned[i]
        vol_ok = vol_spike[i]
        rsi_val = rsi[i]
        price_above_trend = close[i] > trend
        price_below_trend = close[i] < trend
        
        if position == 0:
            # Enter long: volume spike + above trend + RSI not overbought
            if vol_ok and price_above_trend and rsi_val < 70:
                signals[i] = 0.25
                position = 1
            # Enter short: volume spike + below trend + RSI not oversold
            elif vol_ok and price_below_trend and rsi_val > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or RSI overbought
            if not price_above_trend or rsi_val >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or RSI oversold
            if not price_below_trend or rsi_val <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hEMA_VolumeSpike_RSI"
timeframe = "4h"
leverage = 1.0