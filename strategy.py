#!/usr/bin/env python3
"""
Experiment #1159: 6h Camarilla Pivot + 12h Trend + Volume Spike
HYPOTHESIS: Camarilla pivot levels on 6h timeframe identify institutional support/resistance. 
Trend filter from 12h ensures we trade with higher timeframe momentum. Volume spikes (>2x avg) confirm participation. 
In ranging markets (bear/range), we fade at R3/S3 levels for mean reversion. 
In trending markets (bull), we breakout at R4/S4 levels for continuation. 
Designed to work in both bull (breakouts at R4/S4) and bear (mean revert at R3/S3) markets. 
Target: 75-200 total trades over 4 years (19-50/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1159_6h_camarilla_pivot_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # Simple trend: price > previous close = uptrend, < = downtrend
    trend_12h = np.zeros(len(close_12h))
    trend_12h[1:] = np.where(close_12h[1:] > close_12h[:-1], 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === HTF: 1d data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # PP = (H + L + C) / 3
    # R4 = PP + ((H - L) * 1.1 / 2)
    # R3 = PP + ((H - L) * 1.1 / 4)
    # S3 = PP - ((H - L) * 1.1 / 4)
    # S4 = PP - ((H - L) * 1.1 / 2)
    pp = (high_1d + low_1d + close_1d) / 3.0
    r4 = pp + ((high_1d - low_1d) * 1.1 / 2.0)
    r3 = pp + ((high_1d - low_1d) * 1.1 / 4.0)
    s3 = pp - ((high_1d - low_1d) * 1.1 / 4.0)
    s4 = pp - ((high_1d - low_1d) * 1.1 / 2.0)
    
    # Align HTF levels to LTF (6h) with shift(1) for completed bars only
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_12h_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Determine market regime based on 12h trend
            if trend_12h_aligned[i] > 0:  # 12h uptrend -> look for breakouts at R4
                if price > r4_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
            elif trend_12h_aligned[i] < 0:  # 12h downtrend -> look for breakdowns at S4
                if price < s4_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:  # No clear trend -> mean revert at R3/S3 levels
                if price < r3_aligned[i] and price > s3_aligned[i]:
                    # In range, look for reversals at extremes
                    if price <= s3_aligned[i]:  # Near support -> long
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = SIZE
                    elif price >= r3_aligned[i]:  # Near resistance -> short
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = -SIZE
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals