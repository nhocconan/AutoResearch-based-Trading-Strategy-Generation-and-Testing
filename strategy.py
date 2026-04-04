#!/usr/bin/env python3
"""
Experiment #4339: 6h Camarilla Pivot + 12h Trend + Volume Spike
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from 12h timeframe provide institutional support/resistance. 
In ranging markets (ADX<25): fade extreme moves at R3/S3 with volume confirmation. 
In trending markets (ADX>=25): breakout continuation at R4/S4 with volume spike. 
Uses 12h for pivot calculation (more stable than 1d) and 6h for execution. 
Target: 75-150 total trades over 4 years (19-37/year) with position size 0.25.
Works in bull via R4 breakouts, in bear via S4 breakdowns, and in range via R3/S3 reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4339_6h_camarilla_12h_trend_vol_v1"
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
    
    # === Precompute HTF: 12h data for Camarilla pivots and ADX ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 2:
        # Calculate typical price for pivot
        typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
        
        # Camarilla pivot levels (based on previous day's range)
        # R4 = Close + ((High - Low) * 1.1/2)
        # R3 = Close + ((High - Low) * 1.1/4)
        # S3 = Close - ((High - Low) * 1.1/4)
        # S4 = Close - ((High - Low) * 1.1/2)
        prev_close = df_12h['close'].shift(1)
        prev_high = df_12h['high'].shift(1)
        prev_low = df_12h['low'].shift(1)
        prev_range = prev_high - prev_low
        
        r4 = prev_close + (prev_range * 1.1 / 2.0)
        r3 = prev_close + (prev_range * 1.1 / 4.0)
        s3 = prev_close - (prev_range * 1.1 / 4.0)
        s4 = prev_close - (prev_range * 1.1 / 2.0)
        
        # Calculate ADX for regime filter (12h)
        period = 14
        alpha = 1.0 / period
        
        # True Range components
        high_low = df_12h['high'] - df_12h['low']
        high_close = np.abs(df_12h['high'] - df_12h['close'].shift(1))
        low_close = np.abs(df_12h['low'] - df_12h['close'].shift(1))
        tr_12h = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # Directional Movement
        up_move = df_12h['high'].diff()
        down_move = df_12h['low'].diff().mul(-1)
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        atr_12h = np.zeros(len(df_12h))
        plus_dm_smooth = np.zeros(len(df_12h))
        minus_dm_smooth = np.zeros(len(df_12h))
        
        # Initial values
        if len(df_12h) >= period:
            atr_12h[period-1] = np.mean(tr_12h.iloc[:period])
            plus_dm_smooth[period-1] = np.mean(plus_dm.iloc[:period])
            minus_dm_smooth[period-1] = np.mean(minus_dm.iloc[:period])
            
            # Wilder's smoothing
            for i in range(period, len(df_12h)):
                atr_12h[i] = atr_12h[i-1] * (1 - alpha) + alpha * tr_12h.iloc[i]
                plus_dm_smooth[i] = plus_dm_smooth[i-1] * (1 - alpha) + alpha * plus_dm.iloc[i]
                minus_dm_smooth[i] = minus_dm_smooth[i-1] * (1 - alpha) + alpha * minus_dm.iloc[i]
        
        # Avoid division by zero
        plus_di_12h = np.where(atr_12h != 0, 100 * plus_dm_smooth / atr_12h, 0)
        minus_di_12h = np.where(atr_12h != 0, 100 * minus_dm_smooth / atr_12h, 0)
        dx_12h = np.where((plus_di_12h + minus_di_12h) != 0, 
                          100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h), 0)
        
        adx_12h = np.full(len(df_12h), np.nan)
        if len(df_12h) >= 2*period-1:
            adx_12h[2*period-2] = np.nanmean(dx_12h.iloc[period-1:2*period-1])
            for i in range(2*period-1, len(df_12h)):
                adx_12h[i] = adx_12h[i-1] * (1 - alpha) + alpha * dx_12h.iloc[i]
        
        # Align HTF arrays to LTF
        r4_aligned = align_htf_to_ltf(prices, df_12h, r4.values)
        r3_aligned = align_htf_to_ltf(prices, df_12h, r3.values)
        s3_aligned = align_htf_to_ltf(prices, df_12h, s3.values)
        s4_aligned = align_htf_to_ltf(prices, df_12h, s4.values)
        adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    else:
        r4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
        adx_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14, 2*14)  # vol MA, ATR, ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        # Regime filter: ADX < 25 = ranging (mean reversion), ADX >= 25 = trending (breakout)
        ranging_market = adx_12h_aligned[i] < 25.0
        trending_market = adx_12h_aligned[i] >= 25.0
        
        if volume_confirm:
            if ranging_market:
                # Mean reversion at extreme levels (R3/S3)
                long_entry = price <= s3_aligned[i]  # Price at or below S3
                short_entry = price >= r3_aligned[i]  # Price at or above R3
            elif trending_market:
                # Breakout continuation at stronger levels (R4/S4)
                long_entry = price >= r4_aligned[i]  # Price breaks above R4
                short_entry = price <= s4_aligned[i]  # Price breaks below S4
            else:
                long_entry = False
                short_entry = False
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals