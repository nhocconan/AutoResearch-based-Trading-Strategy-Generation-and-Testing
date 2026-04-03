#!/usr/bin/env python3
"""
Experiment #407: 6h Camarilla Pivot + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: 6h Camarilla pivot breakouts at R4/S4 levels with 1d volume confirmation (>2.0x average) 
and 1w trend filter (price > EMA50 on weekly for longs, < EMA50 for shorts) captures strong momentum 
breakouts in both bull and bear markets. Using 6h primary timeframe with 1d/1w HTF filters reduces 
noise and overtrading. Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag 
while maintaining statistical significance.
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
    n = len(close)
    
    # === HTF: 1d data for volume spike calculation ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close for trend filter
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Calculate Camarilla pivot levels from previous day ===
    # Camarilla levels based on previous day's OHLC
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We need to get previous day's OHLC for each 6h bar
    
    # First, resample to daily to get OHLC (but we'll use the HTF data we already have)
    # df_1d already contains daily OHLC
    if len(df_1d) >= 1:
        # Calculate Camarilla levels for each day
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Camarilla R4 and S4 levels
        camarilla_r4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
        camarilla_s4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
        
        # Align to 6h timeframe
        camarilla_r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
        camarilla_s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    else:
        camarilla_r4_1d_aligned = np.full(n, np.nan)
        camarilla_s4_1d_aligned = np.full(n, np.nan)
    
    # === Session filter: 00-23 UTC (trade all hours for 6h timeframe) ===
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Trade all hours for 6h timeframe ---
        hour = hours[i]
        # No session filter for 6h - trade continuously
        
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r4_1d_aligned[i]) or np.isnan(camarilla_s4_1d_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
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
                # Take profit at Camarilla S4 (trailing stop for longs)
                if close[i] <= camarilla_s4_1d_aligned[i]:
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
                # Take profit at Camarilla R4 (trailing stop for shorts)
                if close[i] >= camarilla_r4_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Camarilla R4 with volume confirmation and uptrend
        long_condition = (
            close[i] > camarilla_r4_1d_aligned[i] and  # Breakout above R4
            vol_ratio_1d_aligned[i] > 2.0 and  # Volume spike confirmation
            close[i] > ema_50_1w_aligned[i]   # Price above weekly EMA50 (uptrend)
        )
        
        # Short: Price breaks below Camarilla S4 with volume confirmation and downtrend
        short_condition = (
            close[i] < camarilla_s4_1d_aligned[i] and  # Breakdown below S4
            vol_ratio_1d_aligned[i] > 2.0 and  # Volume spike confirmation
            close[i] < ema_50_1w_aligned[i]   # Price below weekly EMA50 (downtrend)
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