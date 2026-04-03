#!/usr/bin/env python3
"""
Experiment #1551: 6h Camarilla Pivot + Volume Spike + ADX Trend Filter
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) combined with volume confirmation (>2.0x average) and ADX trend filter (>25) capture high-probability reversals and continuations in both bull and bear markets. The 1d timeframe provides pivot structure, while volume spikes confirm institutional participation. ADX ensures we only trade in trending or high-momentum conditions, avoiding choppy markets. Position size 0.25 balances risk and return. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1551_6h_camarilla_pivot_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + Range * 1.1 / 2
    # S3 = Pivot - Range * 1.1 / 2
    # R4 = Pivot + Range * 1.1
    # S4 = Pivot - Range * 1.1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1 / 2.0
    s3_1d = pivot_1d - range_1d * 1.1 / 2.0
    r4_1d = pivot_1d + range_1d * 1.1
    s4_1d = pivot_1d - range_1d * 1.1
    
    # Align pivot levels to 6h timeframe (shifted by 1 for completed bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ADX(14) for trend strength ===
    # ADX calculation: +DM, -DM, TR, then smoothed
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(high[i] - high[i-1], 0) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(low[i-1] - low[i], 0) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    tr[0] = high[0] - low[0]
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(arr, period):
        alpha = 1.0 / period
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.mean(arr[:period])  # seed
        for i in range(period, len(arr)):
            smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    period_adx = 14
    atr_adx = wilders_smoothing(tr, period_adx)
    plus_di = 100 * wilders_smoothing(plus_dm, period_adx) / (atr_adx + 1e-10)
    minus_di = 100 * wilders_smoothing(minus_dm, period_adx) / (atr_adx + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smoothing(dx, period_adx)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, period_adx)  # sufficient for volume MA and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss (using TR as proxy for volatility) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate current ATR for stoploss (using recent TR)
            if i >= 14:
                recent_tr = tr[i-13:i+1]
                current_atr = np.mean(recent_tr)
            else:
                current_atr = np.mean(tr[:i+1]) if i > 0 else tr[0]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * current_atr
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * current_atr
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require ADX > 25 for sufficient trend/momentum
        strong_momentum = adx[i] > 25
        
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if strong_momentum and volume_spike:
            # Mean reversion at R3/S3: price touches extreme level and reverses
            if price <= s3_1d_aligned[i] and price > s4_1d_aligned[i]:  # In S3-S4 zone, potential long
                # Look for rejection: current close > open (bullish candle) and price above S3
                if close[i] > prices["open"].iloc[i] and close[i] > s3_1d_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            elif price >= r3_1d_aligned[i] and price < r4_1d_aligned[i]:  # In R3-R4 zone, potential short
                # Look for rejection: current close < open (bearish candle) and price below R3
                if close[i] < prices["open"].iloc[i] and close[i] < r3_1d_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            # Breakout continuation at R4/S4: price breaks beyond extreme level
            elif price > r4_1d_aligned[i]:  # Break above R4
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < s4_1d_aligned[i]:  # Break below S4
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