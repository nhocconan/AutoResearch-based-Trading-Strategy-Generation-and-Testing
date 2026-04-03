#!/usr/bin/env python3
"""
Experiment #071: 6h Camarilla Pivot + Volume Spike + Regime Filter (ADX)
HYPOTHESIS: Fading at Camarilla R3/S3 levels with volume confirmation and ADX regime filter captures mean reversion in ranging markets while avoiding false breakouts in trending markets. ADX < 20 defines ranging regime where Camarilla levels act as support/resistance. Volume spike (>1.5x) confirms institutional interest at these levels. Discrete sizing (0.25) and ATR(14) stoploss (2.0*ATR). Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_071_6h_camarilla_pivot_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior day
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # Where C = close, H = high, L = low of prior period
    prior_close = df_1d['close'].shift(1).values
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    camarilla_r3 = prior_close + ((prior_high - prior_low) * 1.1 / 4)
    camarilla_s3 = prior_close - ((prior_high - prior_low) * 1.1 / 4)
    camarilla_r4 = prior_close + ((prior_high - prior_low) * 1.1 / 2)
    camarilla_s4 = prior_close - ((prior_high - prior_low) * 1.1 / 2)
    
    # Align to 6h timeframe (shift(1) already applied above for prior day)
    r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_1d = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_1d = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: ADX(14) for regime filter ===
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or np.isnan(r4_1d[i]) or np.isnan(s4_1d[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filter: Only trade in ranging markets (ADX < 20) ---
        ranging_regime = adx[i] < 20
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
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
        if ranging_regime and volume_spike:
            # Long: price at or below S3 (support) with volume spike
            if price <= s3_1d[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price at or above R3 (resistance) with volume spike
            elif price >= r3_1d[i]:
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