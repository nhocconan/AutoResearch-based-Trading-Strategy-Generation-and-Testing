#!/usr/bin/env python3
"""
Experiment #015: 6h Camarilla Pivot + Volume Spike + 1w Trend Filter
HYPOTHESIS: Camarilla pivot levels from daily timeframe identify institutional support/resistance. 
Breakout above R4 or below S4 with volume spike (>2x average) continues trend when aligned with 1w EMA(21) direction.
Fade at R3/S3 with volume spike reverses when counter to weekly trend. 
Works in bull (continuation breaks at R4/S4) and bear (mean reversion at R3/S3 with trend filter).
Target: 75-150 total trades over 4 years (19-37/year). Uses discrete sizing (0.25) to control fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_015_6h_camarilla_pivot_vol_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla Pivot Levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate Camarilla levels from previous day's OHLC
        # Using previous day's data to avoid look-ahead
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_range = prev_high - prev_low
        
        # Camarilla levels
        camarilla_h5 = prev_close + 1.1 * prev_range * 1.1 / 6  # R4
        camarilla_h4 = prev_close + 1.1 * prev_range * 1.0 / 6  # R3
        camarilla_h3 = prev_close + 1.1 * prev_range * 0.5 / 6  # S3 (actually lower)
        camarilla_l3 = prev_close - 1.1 * prev_range * 0.5 / 6  # S3
        camarilla_l4 = prev_close - 1.1 * prev_range * 1.0 / 6  # R4 (actually higher)
        camarilla_l5 = prev_close - 1.1 * prev_range * 1.1 / 6  # S4
        
        # Align to 6h timeframe (forward fill previous day's levels)
        camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    else:
        camarilla_h5_aligned = np.full(n, np.nan)
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
        camarilla_l5_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for EMA(21) Trend Filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        ema_21 = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
        # Trend: price above EMA = bullish, below = bearish
        ema_trend_aligned = ema_21_aligned
    else:
        ema_trend_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_l5_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(ema_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 2.0  # Require strong volume confirmation
        
        # Camarilla levels
        r4 = camarilla_h5_aligned[i]   # Actually R4
        r3 = camarilla_h4_aligned[i]   # Actually R3
        s3 = camarilla_l3_aligned[i]   # Actually S3
        s4 = camarilla_l5_aligned[i]   # Actually S4
        
        ema_val = ema_trend_aligned[i]
        price_above_ema = price > ema_val
        price_below_ema = price < ema_val
        
        # --- Exit Logic: ATR-based stoploss (using 2.0*ATR) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr = np.zeros(i+1)
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr[0] = high[0] - low[0]
                atr_val = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            else:
                atr_val = 0.0
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr_val
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~32 hours on 6h)
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if vol_spike:
            # Long breakout: price > R4 with volume spike AND price above weekly EMA (bullish continuation)
            if price > r4 and price_above_ema:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short breakout: price < S4 with volume spike AND price below weekly EMA (bearish continuation)
            elif price < s4 and price_below_ema:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            # Long mean reversion: price < R3 with volume spike AND price below weekly EMA (fade bearish rally)
            elif price < r3 and price_below_ema:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short mean reversion: price > S3 with volume spike AND price above weekly EMA (fade bullish rally)
            elif price > s3 and price_above_ema:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals