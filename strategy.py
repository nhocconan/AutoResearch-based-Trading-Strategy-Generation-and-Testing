#!/usr/bin/env python3
"""
Experiment #831: 6h Camarilla Pivot + 1d Trend Filter + Volume Spike
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
provide high-probability reversal/continuation zones. In bull markets (1d close > SMA50), 
breakouts at R4/S4 are favored; in bear markets (1d close < SMA50), reversals at R3/S3 
are favored. Volume confirmation (>1.5x average) filters false signals. 
Uses discrete position sizing (0.25). Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_831_6h_camarilla1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate SMA50 on 1d for trend filter
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    # Trend: 1 = bull (close > SMA50), -1 = bear (close < SMA50), 0 = neutral
    trend_1d = np.zeros_like(close_1d)
    trend_1d[50:] = np.where(close_1d[50:] > sma50_1d[50:], 1, 
                             np.where(close_1d[50:] < sma50_1d[50:], -1, 0))
    # Align trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 1.5 * range_1d
    camarilla_r3 = close_1d + 1.1 * range_1d
    camarilla_s3 = close_1d - 1.1 * range_1d
    camarilla_s4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
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
    
    warmup = max(50, 20)  # sufficient for SMA50 and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ratio[i]) or
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
            
            # Optional: time-based exit after 8 bars (~48h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine market regime from 1d trend
            is_bull = trend_1d_aligned[i] > 0
            is_bear = trend_1d_aligned[i] < 0
            
            if is_bull:
                # Bull market: favor breakout continuation at R4/S4
                if price > camarilla_r4_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price < camarilla_s4_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif is_bear:
                # Bear market: favor mean reversion at R3/S3
                if price < camarilla_r3_aligned[i] and price > camarilla_s3_aligned[i]:
                    # In bear market, look for short entries near R3 (resistance)
                    if price < camarilla_r3_aligned[i] * 1.001:  # slight buffer below R3
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = -SIZE
                    else:
                        signals[i] = 0.0
                elif price > camarilla_s3_aligned[i] and price < camarilla_r3_aligned[i]:
                    # In bear market, look for long entries near S3 (support)
                    if price > camarilla_s3_aligned[i] * 0.999:  # slight buffer above S3
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = SIZE
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            else:
                # Neutral market: no clear bias
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals