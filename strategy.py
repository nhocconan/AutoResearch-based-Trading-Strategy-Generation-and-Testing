#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Reversal_v1
Hypothesis: Daily Camarilla pivot reversals (price touches R3/S3 then reverses) capture mean-reversion moves in ranging markets. Uses 1-week EMA50 trend filter to only trade reversals aligned with weekly trend direction. Volume confirmation (>1.5x 20-bar average) filters low-quality signals. ATR-based stoploss (2.0x ATR) controls drawdown. Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag on 1d timeframe. Works in bull/bear by only taking reversals in direction of weekly trend.
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
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of EMA50 (50), ATR (14), volume MA (20)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        trend_val = ema50_1w_aligned[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Calculate Camarilla levels for today (using previous bar's OHLC)
        if i >= 1:
            # Use previous bar's OHLC for Camarilla calculation
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_val = prev_high - prev_low
            
            # Camarilla R3 and S3 levels
            r3 = prev_close + range_val * 1.1 / 4
            s3 = prev_close - range_val * 1.1 / 4
        else:
            r3 = close_val
            s3 = close_val
        
        # Trend filter: price > 1w EMA50 = uptrend, price < 1w EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Reversal conditions: price touches R3/S3 then reverses
        # Long reversal: touches S3 then closes above it (bounce from support)
        long_reversal = (low_val <= s3) and (close_val > s3) and is_uptrend
        # Short reversal: touches R3 then closes below it (rejection from resistance)
        short_reversal = (high_val >= r3) and (close_val < r3) and is_downtrend
        
        # Entry conditions: Reversal + volume confirmation
        long_entry = long_reversal and vol_conf
        short_entry = short_reversal and vol_conf
        
        # Exit conditions: ATR-based stoploss or opposite Camarilla touch
        long_exit = False
        short_exit = False
        if position == 1:
            # Long stoploss: entry price - 2.0 * ATR
            stop_price = entry_price - 2.0 * atr_val
            long_exit = close_val < stop_price or close_val >= r3  # Stop or Camarilla R3 breakout
        elif position == -1:
            # Short stoploss: entry price + 2.0 * ATR
            stop_price = entry_price + 2.0 * atr_val
            short_exit = close_val > stop_price or close_val <= s3  # Stop or Camarilla S3 breakdown
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Approximate entry price for stop calculation
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_Camarilla_Pivot_Reversal_v1"
timeframe = "1d"
leverage = 1.0