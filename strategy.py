#!/usr/bin/env python3
"""
Experiment #3391: 6h Camarilla Pivot + 1d Direction + Volume Spike
HYPOTHESIS: Camarilla pivot levels from 1d (R3/S3 for mean reversion, R4/S4 for breakout) 
capture institutional reaction zones. 1d EMA(50) provides directional filter to avoid 
counter-trend trades. Volume spike (>1.8x 20-period average) confirms participation. 
ATR trailing stop (2.0x) manages risk. Position size 0.25. 
Designed for 6h timeframe to balance trade frequency and capture both bull/bear regimes 
through adaptive pivot logic (mean reversion in range, breakout in trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3391_6h_camarilla1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    def camarilla(high, low, close):
        # Typical price
        tp = (high + low + close) / 3.0
        # Range
        rng = high - low
        # Camarilla levels
        r4 = tp + (rng * 1.1 / 2)
        r3 = tp + (rng * 1.1 / 4)
        s3 = tp - (rng * 1.1 / 4)
        s4 = tp - (rng * 1.1 / 2)
        return r3, r4, s3, s4
    
    r3_1d, r4_1d, s3_1d, s4_1d = camarilla(high_1d, low_1d, close_1d)
    
    # Calculate EMA(50) on 1d close for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF data to 6s timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches opposite Camarilla level (mean reversion/breakout)
                elif price >= r4_1d_aligned[i]:  # Strong breakout - take profit
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches opposite Camarilla level (mean reversion/breakout)
                elif price <= s4_1d_aligned[i]:  # Strong breakdown - take profit
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # 1d EMA trend filter
            price_vs_ema = price - ema_1d_aligned[i]
            
            # Mean reversion entries at R3/S3 (fade extreme moves)
            # Long: price drops to S3 with bullish 1d trend
            if price <= s3_1d_aligned[i] and price_vs_ema > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: price rises to R3 with bearish 1d trend
            elif price >= r3_1d_aligned[i] and price_vs_ema < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            # Breakout entries at R4/S4 (continuation of strong moves)
            # Long: price breaks above R4 with bullish 1d trend
            elif price >= r4_1d_aligned[i] and price_vs_ema > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: price breaks below S4 with bearish 1d trend
            elif price <= s4_1d_aligned[i] and price_vs_ema < 0:
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