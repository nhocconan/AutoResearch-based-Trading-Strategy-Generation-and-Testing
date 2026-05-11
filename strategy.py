#!/usr/bin/env python3
"""
6h_RSI_Trend_Filter
Hypothesis: Use RSI(14) on 6h for momentum with 1d trend filter (EMA50) and volume confirmation.
Long when RSI < 30 (oversold) and price above 1d EMA50 with volume > 1.5x median.
Short when RSI > 70 (overbought) and price below 1d EMA50 with volume > 1.5x median.
Exit when RSI returns to neutral zone (40-60) or opposite extreme.
Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
Target: 15-30 trades/year to avoid fee drag.
"""

name = "6h_RSI_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- RSI(14) on 6h ---
    delta = np.diff(close_6h, prepend=close_6h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Volume Filter: above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_6h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            if position != 0:
                # Simple exit on invalid data
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine 1d trend
        trend_up = close_6h[i] > ema50_1d_aligned[i]
        trend_down = close_6h[i] < ema50_1d_aligned[i]
        
        # Volume filter: above 1.5x median
        vol_ok = volume_6h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume
            if rsi[i] < 30 and trend_up and vol_ok:
                # Long: RSI oversold + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            elif rsi[i] > 70 and trend_down and vol_ok:
                # Short: RSI overbought + 1d downtrend + volume
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Exit conditions
            if position == 1:
                # Exit: RSI returns to neutral (40-60) or becomes overbought
                if rsi[i] >= 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: RSI returns to neutral (40-60) or becomes oversold
                if rsi[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals