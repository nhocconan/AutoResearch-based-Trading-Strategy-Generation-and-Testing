#!/usr/bin/env python3
"""
4h_DonchianBreakout_1dTrend_Volume_Strict
Hypothesis: Price breaks Donchian(20) high/low on 4h with 1d EMA50 trend filter and volume spike >2x median.
Breakouts capture momentum; trend filter avoids counter-trend trades; volume ensures conviction.
Designed for 15-25 trades/year per symbol to minimize fee drag while capturing strong moves.
Works in bull (breaks highs) and bear (breaks lows) via symmetric long/short logic.
"""

name = "4h_DonchianBreakout_1dTrend_Volume_Strict"
timeframe = "4h"
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
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 4h Donchian Channels (20-period high/low) ---
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Filter: spike above 2x median of last 50 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=50, min_periods=20).median().values
    vol_threshold = vol_median * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for Donchian(20) and EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss: 2x ATR(10) from entry
                atr_10 = _calculate_atr(high_4h, low_4h, close_4h, 10)
                if not np.isnan(atr_10[i]):
                    if position == 1 and close_4h[i] <= entry_price - 2.0 * atr_10[i]:
                        signals[i] = 0.0
                        position = 0
                    elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr_10[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        trend_down = close_4h[i] < ema50_1d_aligned[i]
        
        # Volume filter: spike above 2x median
        vol_ok = volume_4h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if close_4h[i] > highest_high[i] and trend_up and vol_ok:
                # Long: price breaks above Donchian high + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif close_4h[i] < lowest_low[i] and trend_down and vol_ok:
                # Short: price breaks below Donchian low + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Update stoploss and check exits
            atr_10 = _calculate_atr(high_4h, low_4h, close_4h, 10)
            if np.isnan(atr_10[i]):
                # Hold position if ATR not ready
                signals[i] = 0.25 if position == 1 else -0.25
                continue
                
            if position == 1:
                # Stoploss: 2x ATR(10)
                if close_4h[i] <= entry_price - 2.0 * atr_10[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or below midpoint of Donchian channel
                elif close_4h[i] <= (highest_high[i] + lowest_low[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss: 2x ATR(10)
                if close_4h[i] >= entry_price + 2.0 * atr_10[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or above midpoint of Donchian channel
                elif close_4h[i] >= (highest_high[i] + lowest_low[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

def _calculate_atr(high, low, close, window):
    """Calculate ATR with proper handling of first value"""
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
    return atr