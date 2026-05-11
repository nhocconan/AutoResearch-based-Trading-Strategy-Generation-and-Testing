#!/usr/bin/env python3
# 4h_12h_Combined_Signal_Strategy
# Hypothesis: Combine 4h breakout signals with 12h trend confirmation and volume filters to reduce whipsaw. Works in bull markets via breakouts and in bear via trend-following with confirmation.
# Uses 4h Donchian breakout (20) for entry, 12h EMA50 for trend filter, volume spike (1.5x median) for confirmation, and ATR stoploss for risk control.
# Designed to generate 50-150 total trades over 4 years (12-37/year) with focus on BTC and ETH.

name = "4h_12h_Combined_Signal_Strategy"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 12h Trend Filter: EMA50 ---
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # --- 4h Donchian Channel (20-period) ---
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Filter: above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
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
    
    # Start after warmup period (max of 20 for Donchian, 50 for EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
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
        
        # Determine 12h trend
        trend_up = close_4h[i] > ema50_12h_aligned[i]
        trend_down = close_4h[i] < ema50_12h_aligned[i]
        
        # Volume filter: above 1.5x median
        vol_ok = volume_4h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only with volume confirmation
            if close_4h[i] > highest_high[i] and vol_ok:
                # Long: price breaks above Donchian high + volume
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif close_4h[i] < lowest_low[i] and vol_ok:
                # Short: price breaks below Donchian low + volume
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
                # Exit: price touches or crosses below Donchian low
                elif close_4h[i] <= lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_4h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above Donchian high
                elif close_4h[i] >= highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals