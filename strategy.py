#!/usr/bin/env python3
"""
Experiment #2727: 6h Camarilla Pivot + Volume Spike + 1d Trend Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) act as institutional support/resistance.
In 6h timeframe: fade at R3/S3 (mean reversion) during low volatility, breakout continuation
at R4/S4 during high volatility. Uses 1d EMA50 for trend filter and volume spike (>2x) for
confirmation. Designed to work in both bull (breakouts) and bear (fades at R3/S3) markets.
Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2727_6h_camarilla_vol_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend and Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate previous day's Camarilla pivot levels (use shift(1) via align_htf_to_ltf)
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    #          S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = typical_price_1d + (range_1d * 1.1 / 2.0)
    r3_1d = typical_price_1d + (range_1d * 1.1 / 4.0)
    s3_1d = typical_price_1d - (range_1d * 1.1 / 4.0)
    s4_1d = typical_price_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (with shift(1) for completed bars only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
        if (np.isnan(trend_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops below S3 (mean reversion target) or breaks above R4 (failed breakout)
                if price < s3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif price > r4_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises above R3 (mean reversion target) or breaks below S4 (failed breakout)
                if price > r3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif price < s4_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average) to avoid false breakouts
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long logic: 
            # - In uptrend: breakout above R4 (continuation)
            # - In downtrend: mean reversion from S3 (fade)
            if trend_bias > 0 and price > r4_1d_aligned[i]:
                # Uptrend breakout continuation
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif trend_bias < 0 and price < s3_1d_aligned[i]:
                # Downtrend fade at S3
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short logic:
            # - In downtrend: breakdown below S4 (continuation)
            # - In uptrend: mean reversion from R3 (fade)
            elif trend_bias < 0 and price < s4_1d_aligned[i]:
                # Downtrend breakdown continuation
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            elif trend_bias > 0 and price > r3_1d_aligned[i]:
                # Uptrend fade at R3
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