#!/usr/bin/env python3
"""
Experiment #2895: 6h Camarilla Pivot Reversal with Weekly Trend Filter
HYPOTHESIS: Camarilla pivot levels from 1d data provide high-probability reversal zones.
At R3/S3: fade (mean reversion) when price reaches extreme levels.
At R4/S4: breakout continuation when price breaks with weekly trend alignment.
Weekly trend (from 1w data) filters for direction: only take longs in weekly uptrend,
shorts in weekly downtrend. Volume confirmation (>1.5x average) ensures momentum.
This strategy captures both mean reversion in ranges and trend continuation in breaks,
adapting to bull/bear/chop regimes. 6s timeframe targets 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2895_6h_camarilla_pivot_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Pivot = (High + Low + Close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    # Camarilla levels
    r3_1d = pivot_1d + range_1d * 1.1 / 2  # R3 = Pivot + 1.1*(Range)/2
    s3_1d = pivot_1d - range_1d * 1.1 / 2  # S3 = Pivot - 1.1*(Range)/2
    r4_1d = pivot_1d + range_1d * 1.1      # R4 = Pivot + 1.1*Range
    s4_1d = pivot_1d - range_1d * 1.1      # S4 = Pivot - 1.1*Range
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === HTF: 1w data for weekly trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend direction
    weekly_ema = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
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
    
    warmup = max(50, 20, 21)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                atr_estimate = (high[i] - low[i]) * 0.5
                if price < highest_since_entry - 2.5 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches opposite Camarilla level (mean reversion target)
                elif price >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                atr_estimate = (high[i] - low[i]) * 0.5
                if price > lowest_since_entry + 2.5 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches opposite Camarilla level (mean reversion target)
                elif price <= s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average)
        volume_ok = vol_ratio[i] > 1.5
        
        if volume_ok:
            # Get weekly trend bias
            weekly_bullish = price > weekly_ema_aligned[i]
            weekly_bearish = price < weekly_ema_aligned[i]
            
            # Fade at R3/S3 (mean reversion) - only in ranging/choppy markets
            # Long fade at S3: price reaches strong support with weekly uptrend bias
            if price <= s3_aligned[i] and weekly_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short fade at R3: price reaches strong resistance with weekly downtrend bias
            elif price >= r3_aligned[i] and weekly_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            # Breakout continuation at R4/S4 - only with weekly trend alignment
            # Long breakout: price breaks above R4 with weekly uptrend
            elif price >= r4_aligned[i] and weekly_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short breakout: price breaks below S4 with weekly downtrend
            elif price <= s4_aligned[i] and weekly_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals