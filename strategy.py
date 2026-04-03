#!/usr/bin/env python3
"""
Experiment #339: 6h Camarilla Pivot + Volume Spike + Trend Filter (12h/1d)

HYPOTHESIS: Camarilla pivot levels on 12h timeframe act as significant support/resistance.
Breakout above R3 or below S3 with volume confirmation (>1.5x average) and aligned with
1d trend (price > EMA50 for longs, < EMA50 for shorts) captures high-probability momentum.
Using 12h for pivot levels and trend filter, 6h for entry timing. Target: 75-150 total trades
over 4 years (19-37/year) to balance opportunity with fee drag minimization.
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 12h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 2:
        # Calculate Camarilla pivots from previous 12h bar (HLC of completed bar)
        h_12h = df_12h['high'].values
        l_12h = df_12h['low'].values
        c_12h = df_12h['close'].values
        
        # Pivot point = (H + L + C) / 3
        pp_12h = (h_12h + l_12h + c_12h) / 3.0
        # Range = H - L
        range_12h = h_12h - l_12h
        
        # Camarilla levels
        r3_12h = pp_12h + range_12h * 1.1 / 4.0
        s3_12h = pp_12h - range_12h * 1.1 / 4.0
        r4_12h = pp_12h + range_12h * 1.1 / 2.0
        s4_12h = pp_12h - range_12h * 1.1 / 2.0
        
        # Align to 6h timeframe (shifted by 1 for completed bar only)
        r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
        s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
        r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
        s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    else:
        # Not enough data - fill with neutral values
        r3_12h_aligned = np.full(n, np.nan)
        s3_12h_aligned = np.full(n, np.nan)
        r4_12h_aligned = np.full(n, np.nan)
        s4_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume Ratio (20-period) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # Default to 1.0 for warmup period
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
        # --- Session Filter: Only trade 08-20 UTC ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: price relative to 1d EMA50 ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
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
                # Take profit at R4 (aggressive target)
                if close[i] >= r4_12h_aligned[i]:
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
                # Take profit at S4 (aggressive target)
                if close[i] <= s4_12h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above R3 with volume confirmation in uptrend
        long_condition = (
            close[i] > r3_12h_aligned[i] and 
            volume_spike and 
            price_above_1d_ema
        )
        
        # Short: Break below S3 with volume confirmation in downtrend
        short_condition = (
            close[i] < s3_12h_aligned[i] and 
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

}