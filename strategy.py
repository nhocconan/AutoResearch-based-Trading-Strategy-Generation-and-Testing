#!/usr/bin/env python3
"""
Experiment #3279: 6h Camarilla Pivot + 12h ADX Trend + Volume Confirmation
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe capture intraday extremes. 
12h ADX > 25 filters for trending markets to avoid false breakouts in ranging conditions. 
Volume > 1.5x 20-period average confirms institutional participation. 
Position size 0.25. Target: 100-200 total trades over 4 years (25-50/year).
Designed to work in bull markets (breakout continuation at R4/S4) and bear markets (mean reversion at R3/S3) by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3279_6h_camarilla_pivot_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h data
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False).mean().values
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        # DX and ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = C + (H - L) * 1.1/2
    # S3 = C - (H - L) * 1.1/2
    # R4 = C + (H - L) * 1.1
    # S4 = C - (H - L) * 1.1
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    prev_range = prev_high - prev_low
    
    r3 = pivot + prev_range * 1.1 / 2.0
    s3 = pivot - prev_range * 1.1 / 2.0
    r4 = pivot + prev_range * 1.1
    s4 = pivot - prev_range * 1.1
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 14)  # sufficient for volume and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Fixed stop at 2*ATR(6h) ---
        if in_position:
            # Calculate ATR(6h) for stoploss
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if not np.isnan(atr_6h[i]):
                if position_side > 0:  # Long
                    if price < entry_price - 2.0 * atr_6h[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:  # Short
                    if price > entry_price + 2.0 * atr_6h[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
            else:
                signals[i] = 0.0
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation and ADX > 25 (trending market)
        volume_ok = vol_ratio[i] > 1.5
        adx_ok = adx_12h_aligned[i] > 25.0
        
        if volume_ok and adx_ok:
            # Mean reversion at extreme levels (R3/S3) - fade the move
            if price >= r3[i] and price < r4[i]:
                # Short at R3 expecting reversion to pivot
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            elif price <= s3[i] and price > s4[i]:
                # Long at S3 expecting reversion to pivot
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Breakout continuation at extreme levels (R4/S4) - go with the move
            elif price > r4[i]:
                # Long breakout above R4
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif price < s4[i]:
                # Short breakdown below S4
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals