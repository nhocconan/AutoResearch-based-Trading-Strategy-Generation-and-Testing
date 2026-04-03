#!/usr/bin/env python3
"""
Experiment #347: 6h Camarilla Pivot + Volume Spike + 1d Trend Filter

HYPOTHESIS: Camarilla pivot levels derived from 1d timeframe provide significant support/resistance 
zones. Trading breakouts beyond R4/S4 with 6h volume confirmation and aligned with 1d trend 
(captured via price vs EMA50) captures high-probability continuation moves. Using 1d for pivot 
calculation and trend filter, 6h for execution minimizes false breakouts while capturing 
multi-day momentum. Target: 75-150 total trades over 4 years (19-37/year) to balance opportunity 
and fee drag. Works in bull (breakouts continue) and bear (breakdowns continue) via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot calculation and trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate Camarilla pivot levels from previous 1d bar
        camarilla_high = np.full(n, np.nan)
        camarilla_low = np.full(n, np.nan)
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
        camarilla_r4 = np.full(n, np.nan)
        camarilla_s4 = np.full(n, np.nan)
        
        # Pivot point = (H + L + C) / 3
        # Camarilla levels:
        # R4 = C + ((H-L) * 1.1/2)
        # R3 = C + ((H-L) * 1.1/4)
        # S3 = C - ((H-L) * 1.1/4)
        # S4 = C - ((H-L) * 1.1/2)
        for i in range(1, len(high_1d)):
            h = high_1d[i-1]
            l = low_1d[i-1]
            c = close_1d[i-1]
            pivot = (h + l + c) / 3.0
            range_hl = h - l
            
            camarilla_high[i] = pivot
            camarilla_r3[i] = c + (range_hl * 1.1 / 4)
            camarilla_s3[i] = c - (range_hl * 1.1 / 4)
            camarilla_r4[i] = c + (range_hl * 1.1 / 2)
            camarilla_s4[i] = c - (range_hl * 1.1 / 2)
        
        # Align 1d arrays to 6h timeframe
        camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
        
        # 1d trend filter: EMA50
        if len(df_1d) >= 50:
            ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
            ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
        else:
            ema_50_1d_aligned = np.full(n, np.nan)
    else:
        camarilla_high_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Trend Filter: Price vs 1d EMA50 ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Camarilla R3 (trailing stop)
                if close[i] >= camarilla_r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Camarilla S3 (trailing stop)
                if close[i] <= camarilla_s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Camarilla R4 with volume confirmation in uptrend
        long_condition = (
            close[i] > camarilla_r4_aligned[i] and 
            volume_spike and 
            price_above_1d_ema
        )
        
        # Short: Break below Camarilla S4 with volume confirmation in downtrend
        short_condition = (
            close[i] < camarilla_s4_aligned[i] and 
            volume_spike and 
            price_below_1d_ema
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals