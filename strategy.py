#!/usr/bin/env python3
"""
Experiment #134: 1h Supertrend(10,3) + 4h EMA(21) trend filter + 1d Camarilla pivot mean reversion
HYPOTHESIS: In 1h timeframe, use 4h EMA21 for trend direction (bull/bear) and 1d Camarilla pivots for mean-reversion entries. Only take longs in 4h uptrend near S3/S4 and shorts in 4h downtrend near R3/R4. Volume confirmation (>1.5x average) filters weak signals. Supertrend(10,3) on 1h provides precise entry timing and automatic stoploss. Discrete size 0.20 minimizes fee churn. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_134_1h_supertrend_4h_ema21_1d_camarilla_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h EMA21 for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    def calculate_camarilla(high, low, close):
        pt = (high + low + close) / 3.0
        rng = high - low
        r3 = pt + rng * 1.1 / 4
        r4 = pt + rng * 1.1 / 2
        s3 = pt - rng * 1.1 / 4
        s4 = pt - rng * 1.1 / 2
        return r3, r4, s3, s4
    
    r3_1d = np.full(len(df_1d), np.nan)
    r4_1d = np.full(len(df_1d), np.nan)
    s3_1d = np.full(len(df_1d), np.nan)
    s4_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i >= 0:
            r3, r4, s3, s4 = calculate_camarilla(
                df_1d['high'].values[i],
                df_1d['low'].values[i],
                df_1d['close'].values[i]
            )
            r3_1d[i] = r3
            r4_1d[i] = r4
            s3_1d[i] = s3
            s4_1d[i] = s4
    
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1h Indicators: Supertrend(10,3) for entry timing and stoploss ===
    # ATR(10)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(supertrend[i]) or
            np.isnan(direction[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Trend Filter: 4h EMA21 ---
        uptrend_4h = price > ema_4h_aligned[i]
        downtrend_4h = price < ema_4h_aligned[i]
        
        # --- Camarilla Pivot Conditions ---
        near_s3 = abs(price - s3_1d_aligned[i]) / price < 0.005
        near_s4 = abs(price - s4_1d_aligned[i]) / price < 0.005
        near_r3 = abs(price - r3_1d_aligned[i]) / price < 0.005
        near_r4 = abs(price - r4_1d_aligned[i]) / price < 0.005
        
        # --- Supertrend Conditions ---
        supertrend_bull = direction[i] == 1 and price > supertrend[i]
        supertrend_bear = direction[i] == -1 and price < supertrend[i]
        
        # --- Exit Logic: Supertrend reversal ---
        if in_position:
            if position_side > 0:  # Long position
                if direction[i] == -1:  # Supertrend turned bearish
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if direction[i] == 1:  # Supertrend turned bullish
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: 4h uptrend + near S3/S4 + Supertrend bull + volume spike
        if (uptrend_4h and (near_s3 or near_s4) and supertrend_bull and volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: 4h downtrend + near R3/R4 + Supertrend bear + volume spike
        elif (downtrend_4h and (near_r3 or near_r4) and supertrend_bear and volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals