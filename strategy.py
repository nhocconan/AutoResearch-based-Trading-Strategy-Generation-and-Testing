#!/usr/bin/env python3
"""
Experiment #5691: 6h Camarilla Pivot Reversal + Volume Spike + ATR Filter
HYPOTHESIS: On 6h timeframe, price reversals at Camarilla R3/S3 levels with 
volume > 2.0x average and ATR > 1.5x ATR(50) capture high-probability mean 
reversion moves. Camarilla pivots derived from 1d OHLC provide institutional 
support/resistance that works in both bull and bear markets. Volume confirms 
reversal strength, ATR filter ensures sufficient volatility. Discrete sizing 
(0.25) minimizes fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5691_6h_camarilla_pivot_reversal_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for Camarilla pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate Camarilla levels from prior day's OHLC
        # Camarilla formula based on yesterday's range
        yesterday_high = pd.Series(df_1d['high']).shift(1).values
        yesterday_low = pd.Series(df_1d['low']).shift(1).values
        yesterday_close = pd.Series(df_1d['close']).shift(1).values
        
        # Avoid look-ahead: use shift(1) for prior day only
        range_ = yesterday_high - yesterday_low
        camarilla_h5 = yesterday_close + range_ * 1.1 / 2  # R4 equivalent
        camarilla_h4 = yesterday_close + range_ * 1.1 / 4  # R3
        camarilla_h3 = yesterday_close + range_ * 1.1 / 6  # R2
        camarilla_l3 = yesterday_close - range_ * 1.1 / 6  # S2
        camarilla_l4 = yesterday_close - range_ * 1.1 / 4  # S3
        camarilla_l5 = yesterday_close - range_ * 1.1 / 2  # S4
        
        # For reversal strategy: fade at H3/L3 (R3/S3), breakout at H4/L4 (R4/S4)
        camarilla_h3 = camarilla_h3  # R3
        camarilla_l3 = camarilla_l3  # S3
        camarilla_h4 = camarilla_h4  # R4
        camarilla_l4 = camarilla_l4  # S4
    else:
        camarilla_h3 = np.full(len(df_1d), np.nan)
        camarilla_l3 = np.full(len(df_1d), np.nan)
        camarilla_h4 = np.full(len(df_1d), np.nan)
        camarilla_l4 = np.full(len(df_1d), np.nan)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR filter for volatility regime ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_regime = atr / np.where(atr_ma > 0, atr_ma, 1)  # Current ATR vs MA
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 50, 1)  # Volume avg, ATR MA, HTF shift
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(volume_ratio[i]) or np.isnan(vol_regime[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR trailing stop (2.0x) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_price = entry_price - 2.0 * atr[i]
                # Exit: stoploss OR price breaks above H4 (breakout continuation)
                if price <= stop_price or price >= camarilla_h4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                stop_price = entry_price + 2.0 * atr[i]
                # Exit: stoploss OR price breaks below L4 (breakout continuation)
                if price >= stop_price or price <= camarilla_l4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Reversal at H3/L3 with volume and volatility filter
        near_h3 = np.abs(price - camarilla_h3_aligned[i]) <= (0.1 * camarilla_h3_aligned[i])
        near_l3 = np.abs(price - camarilla_l3_aligned[i]) <= (0.1 * camarilla_l3_aligned[i])
        volume_confirmed = volume_ratio[i] > 2.0
        vol_filter = vol_regime[i] > 1.5  # Ensure sufficient volatility
        
        # Short at H3 (R3) resistance, Long at L3 (S3) support
        short_setup = near_h3 and volume_confirmed and vol_filter
        long_setup = near_l3 and volume_confirmed and vol_filter
        
        if short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        elif long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        else:
            signals[i] = 0.0
    
    return signals