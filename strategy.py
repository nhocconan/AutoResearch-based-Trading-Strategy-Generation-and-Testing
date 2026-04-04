#!/usr/bin/env python3
"""
Experiment #2311: 6h Donchian Breakout + 1d Weekly Pivot + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 6h with alignment to 1d weekly pivot levels (R4/S4 for continuation, R3/S3 for mean reversion) and volume confirmation work in both bull and bear markets.
- Primary: 6h Donchian(20) breakout above/below prior 20-bar high/low
- HTF: 1d weekly pivot levels (R3, S3, R4, S4) from prior week OHLC
- Logic: 
  * In uptrend (price > 1d EMA50): Long on Donchian breakout above 20-bar high near weekly R4/R3, Short on breakdown below 20-bar low near weekly S3/S4
  * In downtrend (price < 1d EMA50): Short on Donchian breakdown below 20-bar low near weekly S4/S3, Long on breakout above 20-bar high near weekly R3/R4
  * Only trade when price is within 0.5% of weekly pivot level to avoid false breakouts
  * Require volume > 2.0x 20-bar average for confirmation
- Exit: Opposite Donchian level or ATR(14) stop (2*ATR)
- Target: 75-150 total trades over 4 years (19-37/year) - suitable for 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2311_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend and weekly pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === Calculate Weekly Pivot Levels from 1d OHLC ===
    # Group 1d data into weeks (5 trading days per week approx)
    # We'll use rolling window of 5 days to approximate weekly OHLC
    if len(close_1d) >= 5:
        # Weekly high = max of last 5 daily highs
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        # Weekly low = min of last 5 daily lows
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        # Weekly close = last daily close in the week
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).apply(lambda x: x[-1]).values
    else:
        weekly_high = np.full_like(close_1d, np.nan)
        weekly_low = np.full_like(close_1d, np.nan)
        weekly_close = np.full_like(close_1d, np.nan)
    
    # Calculate weekly pivot and ranges
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Weekly Camarilla levels (R3, S3, R4, S4)
    weekly_r4 = weekly_close + weekly_range * 1.1 / 2.0
    weekly_r3 = weekly_close + weekly_range * 1.1 / 4.0
    weekly_s3 = weekly_close - weekly_range * 1.1 / 4.0
    weekly_s4 = weekly_close - weekly_range * 1.1 / 2.0
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    
    # === 6h Indicators: Donchian(20), ATR(14), Volume MA(20) ===
    # Donchian channels: 20-period high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or
            np.isnan(weekly_s3_aligned[i]) or np.isnan(weekly_r4_aligned[i]) or
            np.isnan(weekly_s4_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian low (contrarian exit)
                elif price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian high (contrarian exit)
                elif price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Trend filter: 1d EMA50
        trend_bias = trend_1d_aligned[i]  # 1 for uptrend, -1 for downtrend
        
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Calculate proximity to weekly pivot levels (within 0.5%)
            def is_near_level(price, level, threshold=0.005):
                return abs(price - level) / level < threshold if level > 0 else False
            
            near_r4 = is_near_level(price, weekly_r4_aligned[i])
            near_r3 = is_near_level(price, weekly_r3_aligned[i])
            near_s3 = is_near_level(price, weekly_s3_aligned[i])
            near_s4 = is_near_level(price, weekly_s4_aligned[i])
            
            if trend_bias > 0:  # Uptrend bias
                # Long: Donchian breakout above 20-bar high near weekly R4/R3
                if price >= donchian_high[i] and (near_r4 or near_r3):
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
                # Short: Donchian breakdown below 20-bar low near weekly S3/S4 (counter-trend fade)
                elif price <= donchian_low[i] and (near_s3 or near_s4):
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
            else:  # Downtrend bias
                # Short: Donchian breakdown below 20-bar low near weekly S4/S3
                if price <= donchian_low[i] and (near_s4 or near_s3):
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                # Long: Donchian breakout above 20-bar high near weekly R3/R4 (counter-trend bounce)
                elif price >= donchian_high[i] and (near_r3 or near_r4):
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
    
    return signals