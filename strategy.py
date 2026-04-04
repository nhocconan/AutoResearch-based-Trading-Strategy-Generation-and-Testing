#!/usr/bin/env python3
"""
Experiment #5859: 6h Elder Ray Power + 1d ADX Regime + Volume Confirmation
HYPOTHESIS: Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13. 
Combined with 1d ADX regime filter (ADX>25 = trending, ADX<20 = ranging) and volume confirmation, 
this captures strong momentum moves in trending markets while avoiding false signals in ranging 
markets. Works in bull markets (strong Bull Power with ADX>25) and bear markets (strong Bear Power 
with ADX>25). Targets 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5859_6h_elder_ray_1d_adx_vol_v1"
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
    
    # === HTF: 1d data for ADX regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        # Calculate ADX(14) on 1d data
        # True Range
        tr1 = df_1d['high'] - df_1d['low']
        tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
        tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr_1d = tr.rolling(window=14, min_periods=14).mean()
        
        # Directional Movement
        up_move = df_1d['high'] - df_1d['high'].shift(1)
        down_move = df_1d['low'].shift(1) - df_1d['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed DM and TR
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
        tr_smooth = atr_1d * 14  # Approximation: ATR * period = smoothed TR
        
        # Directional Indicators
        plus_di_1d = 100 * plus_dm_smooth / tr_smooth
        minus_di_1d = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
        adx_1d = dx.rolling(window=14, min_periods=14).mean()
        
        # Align to 6h timeframe
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d.values)
    else:
        adx_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray Power ===
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13 (negative values indicate selling pressure)
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(13, 20, 14, 20, 14)  # EMA13, volume avg, ATR, ADX
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR weakening bull power (price < EMA13) OR ADX weakening (<20)
                if price <= stop_price or price < ema13[i] or adx_1d_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR weakening bear power (price > EMA13) OR ADX weakening (<20)
                if price >= stop_price or price > ema13[i] or adx_1d_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime filter: ADX > 25 indicates trending market
        is_trending = adx_1d_aligned[i] > 25
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Entry conditions: Strong Elder Ray power in direction of trend with volume
        long_setup = is_trending and (bull_power[i] > 0) and volume_confirmed
        short_setup = is_trending and (bear_power[i] < 0) and volume_confirmed
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals