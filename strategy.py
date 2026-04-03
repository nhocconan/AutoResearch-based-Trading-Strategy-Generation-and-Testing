#!/usr/bin/env python3
"""
Experiment #939: 6h Camarilla Pivot + 12h Trend + Volume Spike
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from 12h act as strong support/resistance. 
Fade at R3/S3 when 12h trend is weak (ADX<25), breakout continuation at R4/S4 when 12h trend is strong (ADX>25).
Volume confirmation (>1.5x average) filters false signals. Target: 75-150 total trades over 4 years (19-37/year) on 6h timeframe.
Works in bull/bear via regime-adaptive logic (fade in ranging, breakout in trending).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_939_6h_camarilla_pivot_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot and ADX trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h (using previous 12h bar's OHLC)
    def camarilla_levels(high, low, close):
        # Typical price for pivot
        pivot = (high + low + close) / 3.0
        range_ = high - low
        # Camarilla levels
        r4 = pivot + (range_ * 1.1 / 2)
        r3 = pivot + (range_ * 1.1 / 4)
        s3 = pivot - (range_ * 1.1 / 4)
        s4 = pivot - (range_ * 1.1 / 2)
        return r3, r4, s3, s4
    
    # Calculate for each 12h bar (using that bar's OHLC as the "previous" bar for next period)
    r3_12h = np.zeros(len(close_12h))
    r4_12h = np.zeros(len(close_12h))
    s3_12h = np.zeros(len(close_12h))
    s4_12h = np.zeros(len(close_12h))
    
    for i in range(1, len(close_12h)):  # Start from 1 to use previous bar
        r3, r4, s3, s4 = camarilla_levels(high_12h[i-1], low_12h[i-1], close_12h[i-1])
        r3_12h[i] = r3
        r4_12h[i] = r4
        s3_12h[i] = s3
        s4_12h[i] = s4
    # First bar remains 0 (no previous bar)
    
    # Calculate ADX(14) on 12h for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First TR
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx = np.where(np.isnan(dx), 0, dx)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    # Trend strength: 1 = strong (ADX>25), 0 = weak (ADX<=25)
    trend_strength_12h = np.where(adx_12h > 25, 1, 0)
    
    # Align HTF indicators to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    trend_strength_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_strength_12h)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
    bars_since_entry = 0
    
    warmup = max(20, 20)  # sufficient for volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(trend_strength_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Opposite signal or time-based exit ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price reaches S3 (fade) or S4 breaks (stop)
                if price <= s3_12h_aligned[i] and trend_strength_12h_aligned[i] == 0:
                    exit_signal = True  # Fade exit in ranging market
                elif price < s4_12h_aligned[i]:
                    exit_signal = True  # Stop loss on breakdown
            else:  # Short position
                # Exit if price reaches R3 (fade) or R4 breaks (stop)
                if price >= r3_12h_aligned[i] and trend_strength_12h_aligned[i] == 0:
                    exit_signal = True  # Fade exit in ranging market
                elif price > r4_12h_aligned[i]:
                    exit_signal = True  # Stop loss on breakout
            
            # Time-based exit: max 12 bars (~3d on 6h) to prevent overtrading
            if bars_since_entry > 12:
                exit_signal = True
            
            if exit_signal:
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
            # Fade logic at R3/S3 when trend is weak (ranging market)
            if trend_strength_12h_aligned[i] == 0:  # Weak trend = ranging
                # Short at R3 with volume spike
                if price >= r3_12h_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                # Long at S3 with volume spike
                elif price <= s3_12h_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            # Breakout logic at R4/S4 when trend is strong (trending market)
            else:  # Strong trend
                # Long on breakout above R4 with volume spike
                if price > r4_12h_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short on breakdown below S4 with volume spike
                elif price < s4_12h_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals