# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_Price_Action_Reversal_at_4h_Close_with_Volume
Hypothesis: Price reversal at 4h close (close crosses open of same bar) with volume confirmation and 1d trend filter.
Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend.
Volume confirms conviction. Target: 20-50 trades/year.
"""

name = "4h_Price_Action_Reversal_at_4h_Close_with_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 4h Reversal Signal: close crosses open of same bar ---
    # Bullish reversal: close > open (green bar) after being below open
    # Bearish reversal: close < open (red bar) after being above open
    # We use the relationship: close > open for bullish, close < open for bearish
    # But we need to ensure it's a meaningful move, not just noise
    bullish_reversal = close_4h > open_4h
    bearish_reversal = close_4h < open_4h
    
    # --- Volume Filter: spike above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    vol_ok = volume_4h > vol_threshold
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_4h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        trend_down = close_4h[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume confirmation
            if bullish_reversal[i] and trend_up and vol_ok[i]:
                # Long: bullish reversal + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif bearish_reversal[i] and trend_down and vol_ok[i]:
                # Short: bearish reversal + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_4h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: reversal signal in opposite direction
                elif bearish_reversal[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_4h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: reversal signal in opposite direction
                elif bullish_reversal[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals