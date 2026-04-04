#!/usr/bin/env python3
"""
Experiment #4359: 6h Donchian Breakout + 12h ADX Trend Filter + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 6h aligned with strong 12h trend (ADX > 25) and confirmed by volume spikes (>2x average) capture institutional momentum with reduced false breakouts. The 12h ADX filter ensures we only trade in trending regimes, avoiding whipsaws in ranging markets. Works in bull via upward breakouts with strong uptrend, in bear via downward breakouts with strong downtrend. Volume confirmation filters low-conviction moves. Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4359_6h_donchian20_12h_adx_vol_v1"
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
    
    # === Precompute HTF: 12h ADX for trend strength ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 14:
        # Calculate ADX(14) on 12h data
        # True Range
        tr1 = df_12h['high'].values[1:] - df_12h['low'].values[1:]
        tr2 = np.abs(df_12h['high'].values[1:] - df_12h['close'].values[:-1])
        tr3 = np.abs(df_12h['low'].values[1:] - df_12h['close'].values[:-1])
        tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((df_12h['high'].values[1:] - df_12h['high'].values[:-1]) > 
                          (df_12h['low'].values[:-1] - df_12h['low'].values[1:]),
                          np.maximum(df_12h['high'].values[1:] - df_12h['high'].values[:-1], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        
        dm_minus = np.where((df_12h['low'].values[:-1] - df_12h['low'].values[1:]) > 
                           (df_12h['high'].values[1:] - df_12h['high'].values[:-1]),
                          np.maximum(df_12h['low'].values[:-1] - df_12h['low'].values[1:], 0), 0)
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/14)
        alpha = 1.0 / 14
        tr_14 = np.zeros_like(tr_12h)
        dm_plus_14 = np.zeros_like(dm_plus)
        dm_minus_14 = np.zeros_like(dm_minus)
        
        # Initialize with first 14-period average
        tr_14[13] = np.nansum(tr_12h[1:15])  # sum of first 14 TR values
        dm_plus_14[13] = np.sum(dm_plus[1:15])
        dm_minus_14[13] = np.sum(dm_minus[1:15])
        
        # Wilder's smoothing for the rest
        for i in range(14, len(tr_12h)):
            tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr_12h[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
        
        # DI+ and DI-
        di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
        di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        
        # ADX: Wilder's smoothing of DX
        adx_12h = np.full_like(dx, np.nan)
        if len(dx) >= 14:
            adx_12h[27] = np.nanmean(dx[14:28])  # First ADX after 2*14 periods
            for i in range(28, len(dx)):
                adx_12h[i] = (adx_12h[i-1] * 13 + dx[i]) / 14
        
        # Align to 6h timeframe
        adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    else:
        adx_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = max(20, 20, 14, 28)  # Donchian, vol MA, ATR, ADX warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(adx_12h_aligned[i])):
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
        # Require strong trend (ADX > 25) and volume confirmation (> 2x average)
        strong_trend = adx_12h_aligned[i] > 25
        volume_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Long conditions: upward breakout + strong trend + volume
        long_entry = breakout_up and strong_trend and volume_confirm
        
        # Short conditions: downward breakout + strong trend + volume
        short_entry = breakout_down and strong_trend and volume_confirm
        
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
    
    return signals