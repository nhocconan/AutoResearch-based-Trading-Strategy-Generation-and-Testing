#!/usr/bin/env python3
"""
Experiment #2615: 6h Camarilla pivot + volume spike + weekly trend filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe 
with volume confirmation and weekly trend filter captures institutional order flow. 
Weekly trend ensures alignment with larger market structure, reducing false signals in choppy markets.
Volume spike (>2x average) confirms participation. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2615_6h_camarilla_vol_weekly_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(20) for trend
    ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # R3 = PP + (H - L) * 1.1/4
    # S3 = PP - (H - L) * 1.1/4
    # S4 = PP - (H - L) * 1.1/2
    pp = (high_1d + low_1d + close_1d) / 3.0
    r4 = pp + (high_1d - low_1d) * 1.1 / 2.0
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    s4 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 6h timeframe (shifted by 1 for completed bars only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
    
    warmup = 20  # sufficient for volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1w_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit conditions:
                # 1. Price reaches R4 (take profit)
                # 2. Price drops below S3 (stop loss/reversal)
                # 3. Weekly trend turns bearish
                if price >= r4_aligned[i] or price <= s3_aligned[i] or trend_1w_aligned[i] < 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit conditions:
                # 1. Price reaches S4 (take profit)
                # 2. Price rises above R3 (stop loss/reversal)
                # 3. Weekly trend turns bullish
                if price <= s4_aligned[i] or price >= r3_aligned[i] or trend_1w_aligned[i] > 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price crosses above S3 with bullish weekly trend (mean reversion long)
            if trend_1w_aligned[i] > 0 and price > s3_aligned[i] and close[i-1] <= s3_aligned[i-1]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price crosses below R3 with bearish weekly trend (mean reversion short)
            elif trend_1w_aligned[i] < 0 and price < r3_aligned[i] and close[i-1] >= r3_aligned[i-1]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            # Long breakout: price breaks above R4 with bullish weekly trend
            elif trend_1w_aligned[i] > 0 and price > r4_aligned[i] and close[i-1] <= r4_aligned[i-1]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short breakout: price breaks below S4 with bearish weekly trend
            elif trend_1w_aligned[i] < 0 and price < s4_aligned[i] and close[i-1] >= s4_aligned[i-1]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals