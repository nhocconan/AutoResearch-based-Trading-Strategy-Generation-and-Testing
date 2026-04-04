#!/usr/bin/env python3
"""
Experiment #5470: 1d Donchian(20) breakout + 1w HMA trend filter + volume confirmation
HYPOTHESIS: On daily timeframe, price breaking above/below the 20-period Donchian channel with 
volume > 2.0x average and aligned with the 1-week Hull Moving Average (HMA21) trend 
captures strong momentum moves while avoiding false breakouts in choppy markets. 
The 1-week HMA provides a smoother, more reliable trend filter than daily EMA, reducing 
whipsaws during bear market rallies and bull market corrections. Discrete position sizing 
(0.25) and ATR-based stoploss (2.5x ATR) control risk. Target: 75-200 total trades over 4 years 
(19-50/year) to minimize fee drag while maintaining statistical significance. Works in bull markets 
via breakouts above rising HMA alignment and in bear markets via short breakdowns below falling HMA 
alignment, with volume confirmation ensuring institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5470_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1w data for HMA21 ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        
        # Calculate HMA(21) on weekly data: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        half_period = 21 // 2
        sqrt_period = int(np.sqrt(21))
        
        wma_half = wma(close_1w, half_period)
        wma_full = wma(close_1w, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21 = wma(wma_diff, sqrt_period)
        
        # Pad beginning with NaN to match original length
        hma_21_padded = np.full(len(close_1w), np.nan)
        hma_21_padded[(len(close_1w) - len(hma_21)):] = hma_21
        
        # Align to LTF (1d) with shift(1) for completed bars only
        hma_21w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded) if len(hma_21_padded) > 0 else np.full(n, np.nan)
    else:
        hma_21w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 1d Indicators: ATR(14) for stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 20, 14, 21)  # Donchian, volume avg, ATR warmup, HMA periods
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        # Trade during major sessions: 00-06 UTC (Asia), 07-12 UTC (Europe), 13-20 UTC (US)
        # Avoid 21-23 UTC (low liquidity between sessions)
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(hma_21w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.5 * ATR below highest since entry
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Weekly HMA turns downward (trend weakening)
                if price <= stop_price or price <= donchian_low[i] or hma_21w_aligned[i] < hma_21w_aligned[i-1]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.5 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Weekly HMA turns upward (trend weakening)
                if price >= stop_price or price >= donchian_high[i] or hma_21w_aligned[i] > hma_21w_aligned[i-1]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Weekly HMA trend: rising = bullish, falling = bearish
        hma_rising = hma_21w_aligned[i] > hma_21w_aligned[i-1]
        hma_falling = hma_21w_aligned[i] < hma_21w_aligned[i-1]
        
        # Entry conditions
        if breakout_up and volume_confirmed and hma_rising:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and hma_falling:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals