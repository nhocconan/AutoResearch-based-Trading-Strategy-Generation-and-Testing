#!/usr/bin/env python3
"""
Experiment #5274: 1h Donchian Breakout with 4h Trend Filter and Volume Confirmation
HYPOTHESIS: On 1h timeframe, we use 4h Donchian channels for trend direction and 1h Donchian breakouts for entry timing, filtered by 1d EMA50 regime and volume spikes. This captures momentum moves in both bull and bear markets while avoiding whipsaws. The 4h trend filter ensures we only trade in the direction of the higher timeframe trend, reducing false breakouts. Volume confirmation ensures breakouts have conviction. We use discrete position sizing (0.20) and session filter (08-20 UTC) to minimize fee drag, targeting 15-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5274_1h_donchian_breakout_4h_trend_vol_v1"
timeframe = "1h"
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
    
    # === HTF: 4h data for trend direction (Donchian 20) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 20:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        # 4h Donchian upper/lower (20-period)
        donch_hi_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
        donch_lo_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
        donch_hi_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_hi_4h)
        donch_lo_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_lo_4h)
        # 4h trend: 1 if price > upper, -1 if price < lower, 0 otherwise
        trend_4h = np.where(close[:-1:16] > donch_hi_4h[:-1:16], 1, 
                           np.where(close[:-1:16] < donch_lo_4h[:-1:16], -1, 0))  # Simplified for alignment
        # Properly aligned trend array
        trend_4h_raw = np.where(high_4h > donch_hi_4h, 1, 
                               np.where(low_4h < donch_lo_4h, -1, 0))
        trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_raw)
    else:
        donch_hi_4h_aligned = np.full(n, np.nan)
        donch_lo_4h_aligned = np.full(n, np.nan)
        trend_4h_aligned = np.zeros(n)
    
    # === HTF: 1d data for regime filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Breakout (20) ===
    donch_hi = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donch_lo = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1h Indicators: Volume Spike (20-period avg) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 50)  # Donchian, EMA50 warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(trend_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position when reverse breakout or regime change ---
        if in_position:
            # Exit conditions:
            # 1. Reverse Donchian breakout (opposite direction)
            # 2. Regime change (price crosses 1d EMA50)
            # 3. Trend change (4h trend flips)
            if position_side > 0:  # Long position
                if (price < donch_lo[i]) or (price < ema_50_aligned[i]) or (trend_4h_aligned[i] < 0):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if (price > donch_hi[i]) or (price > ema_50_aligned[i]) or (trend_4h_aligned[i] > 0):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Entry conditions: 
        # 1. Donchian breakout in direction of 4h trend
        # 2. Volume confirmation (vol spike)
        # 3. Regime alignment (price vs 1d EMA50)
        # 4. 4h trend alignment
        bullish_breakout = price > donch_hi[i]
        bearish_breakout = price < donch_lo[i]
        vol_confirmed = vol_spike[i]
        regime_bullish = price > ema_50_aligned[i]
        regime_bearish = price < ema_50_aligned[i]
        trend_bullish = trend_4h_aligned[i] > 0
        trend_bearish = trend_4h_aligned[i] < 0
        
        if bullish_breakout and vol_confirmed and regime_bullish and trend_bullish:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif bearish_breakout and vol_confirmed and regime_bearish and trend_bearish:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals