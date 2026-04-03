#!/usr/bin/env python3
"""
Experiment #091: 6h Camarilla Pivot + Volume Spike + 1d ADX Regime Filter
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) combined with 1d ADX regime filter (ADX>25 for trend, ADX<20 for range) and volume confirmation (>2.0x average) captures high-probability reversals in ranging markets and continuations in trending markets. Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_091_6h_camarilla_pivot_vol_adx_v1"
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
    
    # === 1d Indicators: Camarilla Pivot Levels (based on previous day) ===
    def calculate_camarilla(high, low, close):
        # Camarilla levels based on previous day's range
        range_ = high - low
        camarilla = {
            'R4': close + range_ * 1.1 / 2,
            'R3': close + range_ * 1.1 / 4,
            'R2': close + range_ * 1.1 / 6,
            'R1': close + range_ * 1.1 / 12,
            'PP': (high + low + close) / 3,
            'S1': close - range_ * 1.1 / 12,
            'S2': close - range_ * 1.1 / 6,
            'S3': close - range_ * 1.1 / 4,
            'S4': close - range_ * 1.1 / 2
        }
        return camarilla
    
    # Calculate Camarilla for each 1d bar using previous day's OHLC
    camarilla_levels = []
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_levels.append({'R4': np.nan, 'R3': np.nan, 'S3': np.nan, 'S4': np.nan, 'PP': np.nan})
        else:
            camarilla_levels.append(calculate_camarilla(
                df_1d['high'].values[i-1],
                df_1d['low'].values[i-1],
                df_1d['close'].values[i-1]
            ))
    
    # Extract arrays for each level
    camarilla_R4 = np.array([l['R4'] for l in camarilla_levels])
    camarilla_R3 = np.array([l['R3'] for l in camarilla_levels])
    camarilla_S3 = np.array([l['S3'] for l in camarilla_levels])
    camarilla_S4 = np.array([l['S4'] for l in camarilla_levels])
    camarilla_PP = np.array([l['PP'] for l in camarilla_levels])
    
    # Align to 6h timeframe (shifted by 1 for completed 1d bar only)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    camarilla_PP_aligned = align_htf_to_ltf(prices, df_1d, camarilla_PP)
    
    # === 1d Indicators: ADX(14) for regime filter ===
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First TR is undefined
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    warmup = 50  # Warmup for ADX stability and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_R4_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_S4_aligned[i]) or
            np.isnan(camarilla_PP_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        camarilla_R4 = camarilla_R4_aligned[i]
        camarilla_R3 = camarilla_R3_aligned[i]
        camarilla_S3 = camarilla_S3_aligned[i]
        camarilla_S4 = camarilla_S4_aligned[i]
        camarilla_PP = camarilla_PP_aligned[i]
        adx = adx_1d_aligned[i]
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Regime Classification based on ADX ---
        is_trending = adx > 25
        is_ranging = adx < 20
        # Transition zone (20 <= ADX <= 25) - no clear regime, stay flat
        
        # --- Entry Logic ---
        long_signal = False
        short_signal = False
        
        if is_ranging:
            # In ranging market: mean reversion at extreme Camarilla levels (S3/R3)
            # Long when price touches/below S3 with volume spike
            if price <= camarilla_S3 and volume_spike:
                long_signal = True
            # Short when price touches/above R3 with volume spike
            if price >= camarilla_R3 and volume_spike:
                short_signal = True
        elif is_trending:
            # In trending market: breakout continuation at extreme levels (S4/R4)
            # Long when price breaks above R4 with volume spike (bullish continuation)
            if price >= camarilla_R4 and volume_spike:
                long_signal = True
            # Short when price breaks below S4 with volume spike (bearish continuation)
            if price <= camarilla_S4 and volume_spike:
                short_signal = True
        # In transition zone (20 <= ADX <= 25): no signals
        
        # --- Assign Signals ---
        if long_signal and not short_signal:
            signals[i] = SIZE
        elif short_signal and not long_signal:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals