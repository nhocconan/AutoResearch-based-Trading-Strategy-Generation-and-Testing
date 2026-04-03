#!/usr/bin/env python3
"""
Experiment #1771: 6h Camarilla Pivot + 1d Trend + Volume Spike
HYPOTHESIS: 6h price rejecting Camarilla R3/S3 levels with 1d trend alignment and volume spike (>1.5x) captures reversal in ranging markets and continuation in trending markets. Uses 1d trend filter to avoid counter-trend entries. Position size 0.25 balances return/drawdown. Target: 75-200 total trades over 4 years (19-50/year) via tight Camarilla level confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1771_6h_camarilla_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(50) for trend
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Camarilla Pivot from previous day ===
    # Camarilla levels based on previous 1d OHLC
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)
    
    # Map 6h bars to previous 1d bar (using index mapping)
    # Since open_time is datetime64, we can use date for mapping
    dates_6h = pd.to_datetime(prices["open_time"]).date
    df_1d_index = pd.DatetimeIndex(df_1d.index) if hasattr(df_1d, 'index') else pd.DatetimeIndex(df_1d['open_time'])
    dates_1d = df_1d_index.date
    
    # Create mapping from 6h bar to previous 1d bar's OHLC
    prev_high_1d = np.full(n, np.nan)
    prev_low_1d = np.full(n, np.nan)
    prev_close_1d = np.full(n, np.nan)
    
    j = 0
    for i in range(n):
        # Advance j to the 1d bar that is strictly before current 6h bar's date
        while j < len(dates_1d) and dates_1d[j] <= dates_6h[i]:
            j += 1
        # j-1 is the last 1d bar before current 6h bar
        if j > 0:
            prev_high_1d[i] = high_1d[j-1]
            prev_low_1d[i] = low_1d[j-1]
            prev_close_1d[i] = close_1d[j-1]
    
    # Calculate Camarilla levels: based on previous day's range
    camarilla_high = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 12  # R3
    camarilla_low = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 12   # S3
    camarilla_close = prev_close_1d
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for volume MA and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment
        trend_following = trend_1d_aligned[i] != 0  # Should always be ±1
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if trend_following and volume_spike:
            # Fade at R3/S3: price rejects extreme levels
            if price < camarilla_high[i] and price > camarilla_low[i]:
                # Inside Camarilla H-L: look for rejection at levels
                if abs(price - camarilla_high[i]) < 0.001 * camarilla_high[i]:  # Near R3
                    if trend_1d_aligned[i] < 0:  # Only short in downtrend
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = -SIZE
                elif abs(price - camarilla_low[i]) < 0.001 * camarilla_low[i]:  # Near S3
                    if trend_1d_aligned[i] > 0:  # Only long in uptrend
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = SIZE
            else:
                # Breakout: price breaks above R3 or below S3 with trend
                if price > camarilla_high[i] and trend_1d_aligned[i] > 0:  # Uptrend breakout
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price < camarilla_low[i] and trend_1d_aligned[i] < 0:  # Downtrend breakdown
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals