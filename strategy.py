#!/usr/bin/env python3
"""
Experiment #419: 6h Camarilla Pivot + Volume Spike + 12h Trend Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with 12h volume confirmation (>2.0x average) and 12h trend filter (price > EMA50 
on 12h) captures high-probability reversals and continuations. The 6h timeframe balances 
trade frequency and noise reduction, while 12h HTF filters ensure alignment with higher 
timeframe structure. Target: 75-150 total trades over 4 years (19-38/year) to minimize 
fee drag while maintaining statistical significance. Works in bull markets via R4/S4 
breakouts and in bear markets via R3/S3 mean reversion.
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
    
    # === HTF: 12h data for volume spike and trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # Calculate EMA(50) on 12h close for trend filter
    if len(df_12h) >= 50:
        close_12h = df_12h['close'].values
        ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    else:
        ema_50_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Calculate Camarilla pivot levels from previous bar ===
    if n >= 2:
        # Camarilla levels based on previous bar's range
        close_prev = close[:-1]
        high_prev = high[:-1]
        low_prev = low[:-1]
        range_prev = high_prev - low_prev
        
        # Calculate levels for current bar (based on previous bar)
        camarilla_h4 = close_prev + range_prev * 1.1/2
        camarilla_l4 = close_prev - range_prev * 1.1/2
        camarilla_h3 = close_prev + range_prev * 1.1/4
        camarilla_l3 = close_prev - range_prev * 1.1/4
        camarilla_h2 = close_prev + range_prev * 1.1/6
        camarilla_l2 = close_prev - range_prev * 1.1/6
        camarilla_h1 = close_prev + range_prev * 1.1/12
        camarilla_l1 = close_prev - range_prev * 1.1/12
        
        # Shift to align with current bar (levels from previous bar)
        camarilla_h4 = np.concatenate([np.array([np.nan]), camarilla_h4])
        camarilla_l4 = np.concatenate([np.array([np.nan]), camarilla_l4])
        camarilla_h3 = np.concatenate([np.array([np.nan]), camarilla_h3])
        camarilla_l3 = np.concatenate([np.array([np.nan]), camarilla_l3])
        camarilla_h2 = np.concatenate([np.array([np.nan]), camarilla_h2])
        camarilla_l2 = np.concatenate([np.array([np.nan]), camarilla_l2])
        camarilla_h1 = np.concatenate([np.array([np.nan]), camarilla_h1])
        camarilla_l1 = np.concatenate([np.array([np.nan]), camarilla_l1])
    else:
        camarilla_h4 = camarilla_l4 = camarilla_h3 = camarilla_l3 = np.full(n, np.nan)
        camarilla_h2 = camarilla_l2 = camarilla_h1 = camarilla_l1 = np.full(n, np.nan)
    
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
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i])):
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
                # Take profit at Camarilla L4 (for mean reversion) or H4 (for breakout)
                if position_side == 1 and close[i] <= camarilla_l4[i]:
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
                # Take profit at Camarilla H4 (for mean reversion) or L4 (for breakout)
                if position_side == -1 and close[i] >= camarilla_h4[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Mean reversion long: Price touches S3 with volume confirmation and uptrend bias
        mean_rev_long = (
            low[i] <= camarilla_l3[i] and  # Touches or breaks S3
            vol_ratio_12h_aligned[i] > 2.0 and  # Volume spike confirmation
            close[i] > ema_50_12h_aligned[i]   # Price above 12h EMA50 (uptrend bias)
        )
        
        # Mean reversion short: Price touches R3 with volume confirmation and downtrend bias
        mean_rev_short = (
            high[i] >= camarilla_h3[i] and  # Touches or breaks R3
            vol_ratio_12h_aligned[i] > 2.0 and  # Volume spike confirmation
            close[i] < ema_50_12h_aligned[i]   # Price below 12h EMA50 (downtrend bias)
        )
        
        # Breakout long: Price breaks above R4 with volume confirmation and uptrend
        breakout_long = (
            close[i] > camarilla_h4[i] and  # Breaks above R4
            vol_ratio_12h_aligned[i] > 2.0 and  # Volume spike confirmation
            close[i] > ema_50_12h_aligned[i]   # Price above 12h EMA50 (uptrend)
        )
        
        # Breakout short: Price breaks below S4 with volume confirmation and downtrend
        breakout_short = (
            close[i] < camarilla_l4[i] and  # Breaks below S4
            vol_ratio_12h_aligned[i] > 2.0 and  # Volume spike confirmation
            close[i] < ema_50_12h_aligned[i]   # Price below 12h EMA50 (downtrend)
        )
        
        if mean_rev_long or breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif mean_rev_short or breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals