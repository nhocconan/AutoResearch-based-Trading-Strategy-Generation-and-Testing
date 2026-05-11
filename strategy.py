#!/usr/bin/env python3
"""
6h_Keltner_Breakout_1dTrend_Volume
Hypothesis: Price breaking above/below Keltner Channel upper/lower bands on 6h, filtered by 1d EMA34 trend and volume spike (2x median). Keltner Channel (EMA20 + ATR*2) adapts to volatility, reducing false breakouts in chop. Trend filter from daily timeframe ensures alignment with longer-term momentum. Volume confirms conviction. Designed to work in bull (uptrend breaks) and bear (downtrend breaks). Target: 20-50 trades/year to avoid fee drag.
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
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 6h Keltner Channel (EMA20 + ATR*2) ---
    ema20_6h = pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range
    tr1 = np.abs(high_6h - low_6h)
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_6h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values  # ATR(10) for sensitivity
    
    kc_upper = ema20_6h + 2 * atr_6h
    kc_lower = ema20_6h - 2 * atr_6h
    
    # --- Volume Filter: spike above 2x median of last 20 periods ---
    vol_median = pd.Series(volume_6h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA20 and EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_6h[i] <= entry_price - 2.0 * atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 2.0 * atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_6h[i] > ema34_1d_aligned[i]
        trend_down = close_6h[i] < ema34_1d_aligned[i]
        
        # Volume filter: spike above 2x median
        vol_ok = volume_6h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if close_6h[i] > kc_upper[i] and trend_up and vol_ok:
                # Long: price breaks above Keltner upper + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            elif close_6h[i] < kc_lower[i] and trend_down and vol_ok:
                # Short: price breaks below Keltner lower + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_6h[i] <= entry_price - 2.0 * atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below Keltner lower
                elif close_6h[i] <= kc_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_6h[i] >= entry_price + 2.0 * atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above Keltner upper
                elif close_6h[i] >= kc_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals