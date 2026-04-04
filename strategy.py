#!/usr/bin/env python3
"""
Experiment #4391: 6h Camarilla Pivot Reversal + 1d Volume Spike + ATR Regime Filter
HYPOTHESIS: At 6h timeframe, price reversing from Camarilla R3/S3 levels with 1d volume confirmation (>2.0x average) and ADX < 25 (low volatility regime) captures mean-reversion bounces in ranging markets while avoiding false breakouts. In bear markets, the same logic applies for short reversals at R3. Uses tight entry conditions to target 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4391_6h_camarilla1d_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1d data for Camarilla pivots and volume MA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Typical price for 1d: (H+L+C)/3
        typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
        # Calculate pivot and Camarilla levels from previous 1d bar
        pivot = typical_price.shift(1).values
        range_1d = (df_1d['high'] - df_1d['low']).shift(1).values
        # Camarilla R3/S3 = pivot ± 1.1 * range/2
        r3 = pivot + 1.1 * range_1d / 2.0
        s3 = pivot - 1.1 * range_1d / 2.0
        # Volume MA(20) for 1d
        vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.ones(len(df_1d))
        vol_ratio_1d[20:] = df_1d['volume'].values[20:] / vol_ma_1d[20:]
        # Align to 6h timeframe with shift(1) for completed bars only
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        vol_ratio_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for volatility regime ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.ones(n)
    atr_ratio[50:] = atr[50:] / atr_ma[50:]
    
    # === 6h Indicators: ADX(14) for trend strength ===
    # +DM, -DM calculation
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # True range already calculated as 'tr'
    # Smoothed values
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(np.concatenate([[np.nan], plus_dm])).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(np.concatenate([[np.nan], minus_dm])).ewm(span=14, min_periods=14, adjust=False).mean().values
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr_14
    minus_di = 100 * minus_dm_smooth / atr_14
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, 20, 14)  # ATR MA, volume MA, ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr_ratio[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse at opposite Camarilla level or ATR stop ---
        if in_position:
            # Exit if price reaches opposite Camarilla level (mean reversion complete)
            if position_side > 0:  # Long - exit at S3
                if price <= s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short - exit at R3
                if price >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: 1d volume > 2.0x average
        volume_confirm = vol_ratio_1d_aligned[i] > 2.0
        # Low volatility regime: ATR ratio < 1.2 (avoid high volatility breakouts)
        vol_regime = atr_ratio[i] < 1.2
        # Low trend strength: ADX < 25 (ranging market)
        trend_filter = adx[i] < 25
        
        # Long entry: price at or below S3 with volume confirmation in low vol ranging market
        long_entry = (price <= s3_aligned[i]) and volume_confirm and vol_regime and trend_filter
        # Short entry: price at or above R3 with volume confirmation in low vol ranging market
        short_entry = (price >= r3_aligned[i]) and volume_confirm and vol_regime and trend_filter
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals