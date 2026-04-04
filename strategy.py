#!/usr/bin/env python3
"""
Experiment #3527: 6h Camarilla Pivot Fade/Breakout + 1d Trend Filter + Volume Spike
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with 1d EMA50 trend filter and volume spike confirmation captures 
institutional order flow. In ranging markets (price between R3-S3), fade extremes. 
In trending markets (price outside R3-S3), breakout continuation works. 
Volume spike confirms participation. Works in bull/bear via adaptive logic.
Target: 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3527_6h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === HTF: 1d data for Camarilla pivot levels (prior day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's OHLC for Camarilla calculation
    prior_high = np.concatenate([[np.nan], high_1d[:-1]])
    prior_low = np.concatenate([[np.nan], low_1d[:-1]])
    prior_close = np.concatenate([[np.nan], close_1d[:-1]])
    prior_open = np.concatenate([[np.nan], df_1d['open'].values[:-1]])  # though not used in classic Camarilla
    
    # Camarilla pivot point (classic: (H+L+C)/3)
    pivot = (prior_high + prior_low + prior_close) / 3.0
    # Range
    rang = prior_high - prior_low
    
    # Resistance levels
    r1 = pivot + (rang * 1.1 / 12)
    r2 = pivot + (rang * 1.1 / 6)
    r3 = pivot + (rang * 1.1 / 4)
    r4 = pivot + (rang * 1.1 / 2)
    # Support levels
    s1 = pivot - (rang * 1.1 / 12)
    s2 = pivot - (rang * 1.1 / 6)
    s3 = pivot - (rang * 1.1 / 4)
    s4 = pivot - (rang * 1.1 / 2)
    
    # Align all Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and stoploss ===
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
    
    warmup = max(50, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long
                # Stoploss: 2.5*ATR below entry
                if price < entry_price - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Take profit: exit at opposite Camarilla level
                elif price >= r3_aligned[i]:  # Exit long at R3 (mean reversion) or R4 (breakout)
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Stoploss: 2.5*ATR above entry
                if price > entry_price + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Take profit: exit at opposite Camarilla level
                elif price <= s3_aligned[i]:  # Exit short at S3 (mean reversion) or S4 (breakout)
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
            # Determine market regime: 
            # If price between R3-S3 -> ranging (mean revert at extremes)
            # If price > R3 or < S3 -> trending (breakout continuation)
            in_range = (price <= r3_aligned[i]) and (price >= s3_aligned[i])
            
            if in_range:
                # Ranging market: fade at R3/S3 extremes
                # Short at R3 with stop above R4
                if price >= r3_aligned[i] and price < r4_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
                # Long at S3 with stop below S4
                elif price <= s3_aligned[i] and price > s4_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
            else:
                # Trending market: breakout continuation
                # Use 1d EMA50 as trend filter
                uptrend = price > ema50_1d_aligned[i]
                downtrend = price < ema50_1d_aligned[i]
                
                # Long breakout: price breaks above R4 with uptrend bias
                if price > r4_aligned[i] and uptrend:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                # Short breakout: price breaks below S4 with downtrend bias
                elif price < s4_aligned[i] and downtrend:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals