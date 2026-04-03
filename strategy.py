#!/usr/bin/env python3
"""
Experiment #1739: 6h Camarilla Pivot + Volume Spike + Regime Filter
HYPOTHESIS: Camarilla pivot levels from 1d timeframe provide institutional support/resistance. Fade at R3/S3 levels with volume confirmation (>1.8x average) and trend filter from 12h timeframe (price > EMA50 for longs, < EMA50 for shorts). This strategy captures mean reversion at extreme levels while avoiding counter-trend trades. Position size 0.25 balances return and drawdown. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1739_6h_camarilla_pivot_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 12h data for regime filter (EMA50 trend) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r4 = np.zeros(len(close_1d))
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_s4 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_r4[i] = camarilla_r3[i] = camarilla_s3[i] = camarilla_s4[i] = np.nan
            continue
        # Previous day's range
        range_1d = high_1d[i-1] - low_1d[i-1]
        camarilla_r4[i] = close_1d[i-1] + range_1d * 1.1 / 2
        camarilla_r3[i] = close_1d[i-1] + range_1d * 1.1 / 4
        camarilla_s3[i] = close_1d[i-1] - range_1d * 1.1 / 4
        camarilla_s4[i] = close_1d[i-1] - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
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
        if (np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or stoploss ---
        if in_position:
            # Exit conditions: reverse signal or price moves beyond S4/R4 (failed fade)
            if position_side > 0:  # Long position
                if (price < camarilla_s3_aligned[i] or  # Reached S3 (target)
                    price > camarilla_r4_aligned[i] or  # Broke above R4 (stoploss)
                    trend_12h_aligned[i] < 0):          # Trend changed to downtrend
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if (price > camarilla_r3_aligned[i] or  # Reached R3 (target)
                    price < camarilla_s4_aligned[i] or  # Broke below S4 (stoploss)
                    trend_12h_aligned[i] > 0):          # Trend changed to uptrend
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Fade at extreme Camarilla levels with trend alignment
            # Long: price at S3 level with uptrend regime
            if (abs(price - camarilla_s3_aligned[i]) < camarilla_r4_aligned[i] * 0.001 and  # Near S3
                trend_12h_aligned[i] > 0):  # Uptrend regime
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short: price at R3 level with downtrend regime
            elif (abs(price - camarilla_r3_aligned[i]) < camarilla_r4_aligned[i] * 0.001 and  # Near R3
                  trend_12h_aligned[i] < 0):  # Downtrend regime
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals