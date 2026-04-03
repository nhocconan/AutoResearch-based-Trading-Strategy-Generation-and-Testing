#!/usr/bin/env python3
"""
Experiment #815: 6h Camarilla Pivot + 1w Trend Filter + Volume Spike
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from daily data act as strong intraday support/resistance. 
Fade at R3/S3 (mean reversion) when 1w trend is opposing, breakout continuation at R4/S4 when 1w trend aligns.
Volume spike (>2.0x average) confirms conviction. Works in bull/bear: 1w trend filter ensures we only take 
breakouts in direction of higher timeframe trend, and fades against the trend. Discrete sizing 0.25. 
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_815_6h_camarilla_pivot_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    def camarilla_levels(high, low, close):
        # Classic Camarilla: R4 = close + ((high-low) * 1.1/2), R3 = close + ((high-low) * 1.1/4)
        # S3 = close - ((high-low) * 1.1/4), S4 = close - ((high-low) * 1.1/2)
        range_hl = high - low
        r4 = close + (range_hl * 1.1 / 2)
        r3 = close + (range_hl * 1.1 / 4)
        s3 = close - (range_hl * 1.1 / 4)
        s4 = close - (range_hl * 1.1 / 2)
        return r4, r3, s3, s4
    
    r4_1d, r3_1d, s3_1d, s4_1d = camarilla_levels(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA(21) on 1w for trend
    ema_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    # Trend: 1 = rising (price > ema), -1 = falling (price < ema), 0 = neutral
    trend_1w = np.zeros_like(close_1w)
    trend_1w[21:] = np.where(close_1w[21:] > ema_1w[21:], 1, 
                              np.where(close_1w[21:] < ema_1w[21:], -1, 0))
    # Align trend to 6h timeframe
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
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
    
    warmup = max(20, 21)  # sufficient for volume MA and 1w EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(trend_1w_aligned[i]) or
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
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Fade at R3/S3: mean reversion when 1w trend is opposing
            # Long fade at S3: price < S3 AND 1w trend is falling (-1) -> expect bounce up
            if price < s3_1d_aligned[i] and trend_1w_aligned[i] < 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short fade at R3: price > R3 AND 1w trend is rising (1) -> expect pullback down
            elif price > r3_1d_aligned[i] and trend_1w_aligned[i] > 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            # Breakout continuation at R4/S4: breakout when 1w trend aligns
            # Long breakout: price > R4 AND 1w trend is rising (1)
            elif price > r4_1d_aligned[i] and trend_1w_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short breakout: price < S4 AND 1w trend is falling (-1)
            elif price < s4_1d_aligned[i] and trend_1w_aligned[i] < 0:
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

</think>