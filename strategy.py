#!/usr/bin/env python3
"""
6h_Keltner_Breakout_1dTrend_Volume
Hypothesis: Price breaking above/below 2xATR Keltner channels on 6h, filtered by 1d EMA50 trend and volume spike (2x median). 
Keltner channels adapt to volatility, providing dynamic support/resistance. Trend filter ensures trades align with higher timeframe momentum. 
Volume conviction filters out false breakouts. Works in bull via uptrend breaks above upper channel, in bear via downtrend breaks below lower channel. 
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
"""

name = "6h_Keltner_Breakout_1dTrend_Volume"
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
    
    # --- 6h Keltner Channel (2x ATR) ---
    # True Range
    tr1 = np.abs(high_6h - low_6h)
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Exponential Moving Average (20-period) as middle line
    ema20 = pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Upper and Lower Keltner Bands
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr
    
    # --- Volume Filter: spike above 2x median of last 30 periods ---
    vol_median = pd.Series(volume_6h).rolling(window=30, min_periods=15).median().values
    vol_threshold = vol_median * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA20 and EMA50_1d
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_6h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_6h[i] > ema50_1d_aligned[i]
        trend_down = close_6h[i] < ema50_1d_aligned[i]
        
        # Volume filter: spike above 2x median
        vol_ok = volume_6h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if close_6h[i] > keltner_upper[i] and trend_up and vol_ok:
                # Long: price breaks above upper Keltner + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            elif close_6h[i] < keltner_lower[i] and trend_down and vol_ok:
                # Short: price breaks below lower Keltner + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_6h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below middle line (EMA20)
                elif close_6h[i] <= ema20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_6h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above middle line (EMA20)
                elif close_6h[i] >= ema20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals