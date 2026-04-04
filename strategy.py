#!/usr/bin/env python3
"""
Experiment #3291: 6h Camarilla Pivot + 1d ADX Regime + Volume Spike
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) combined with 1d ADX regime filter (ADX>25 = trend, ADX<20 = range) and volume confirmation (>1.5x average) captures high-probability entries in both bull and bear markets. In ranging regimes (ADX<20), fade extreme levels (R3/S3). In trending regimes (ADX>25), breakout continuation at R4/S4. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3291_6h_camarilla_pivot_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot and ADX (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + Range * 1.1/2
    # S3 = Pivot - Range * 1.1/2
    # R4 = Pivot + Range * 1.1
    # S4 = Pivot - Range * 1.1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1 / 2.0
    s3_1d = pivot_1d - range_1d * 1.1 / 2.0
    r4_1d = pivot_1d + range_1d * 1.1
    s4_1d = pivot_1d - range_1d * 1.1
    
    # Align Camarilla levels to 6h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        # Smoothed TR, DM+
        tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        dm_plus_period = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
        dm_minus_period = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
        # Directional Indicators
        di_plus = 100 * dm_plus_period / tr_period
        di_minus = 100 * dm_minus_period / tr_period
        # DX
        dx = np.abs(di_plus - di_minus) / (np.abs(di_plus) + np.abs(di_minus)) * 100
        # ADX
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 14, 30)  # sufficient for volume, ATR, ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss at 2.0*ATR ---
        if in_position:
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
            if adx_1d_aligned[i] > 25:  # Trending regime
                # Breakout continuation: buy at R4, sell at S4
                if price > r4_1d_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = price
                    signals[i] = SIZE
                elif price < s4_1d_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = price
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif adx_1d_aligned[i] < 20:  # Ranging regime
                # Mean reversion: sell at R3, buy at S3
                if price > r3_1d_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = price
                    signals[i] = -SIZE
                elif price < s3_1d_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = price
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
            else:  # Transition regime (20 <= ADX <= 25) - no trade
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals