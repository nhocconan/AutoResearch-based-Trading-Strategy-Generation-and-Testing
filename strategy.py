#!/usr/bin/env python3
"""
Experiment #1659: 6h Camarilla Pivot Breakout + 12h Volume Spike + 1d Trend Filter
HYPOTHESIS: Camarilla pivot levels from daily timeframe provide institutional support/resistance. Breakouts above R4 or below S4 with 12h volume confirmation (>2x average) and aligned 1d trend capture strong momentum moves. This strategy targets low-frequency, high-conviction breakouts suitable for 6h timeframe, aiming for 50-150 total trades over 4 years (12-37/year). Works in both bull and bear markets by trading with the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1659_6h_camarilla_breakout_12h_vol_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume MA (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # EMA(50) for 1d trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Camarilla Pivot Levels from previous 1d bar ===
    # Need previous day's OHLC for Camarilla calculation
    # Since we're on 6h timeframe, we need to get daily OHLC from 1d data
    # We'll use the 1d data we already loaded to calculate pivots
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    camarilla_r4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_s4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (use previous completed 1d bar)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
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
    
    warmup = 50  # sufficient for EMA50 and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry (wider for 6h)
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume spike: current 6h volume > 2x 12h volume MA
        volume_spike = volume[i] > 2.0 * vol_ma_12h_aligned[i]
        
        # 1d trend filter
        trend_filter = trend_1d_aligned[i]
        
        if volume_spike and trend_filter != 0:
            # Breakout above R4 in uptrend
            if price > r4_1d_aligned[i] and trend_filter > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Breakdown below S4 in downtrend
            elif price < s4_1d_aligned[i] and trend_filter < 0:
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