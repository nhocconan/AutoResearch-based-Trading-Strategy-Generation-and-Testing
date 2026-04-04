#!/usr/bin/env python3
"""
Experiment #5488: 12h Donchian(20) breakout + 1w HMA21 trend + volume confirmation
HYPOTHESIS: On 12h timeframe, price breaking above/below the 20-period Donchian channel with 
volume > 2.0x average and aligned with 1-week HMA21 trend captures strong momentum moves 
while avoiding false breakouts. The 1w HMA21 provides higher timeframe trend filter (more 
responsive than EMA, less lag than SMA), reducing whipsaws in both bull and bear markets. 
Discrete position sizing (0.25) and ATR-based stoploss (2.0x ATR) control risk. 
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while maintaining 
statistical significance. Works in bull markets via breakouts above rising HMA21 alignment and 
in bear markets via short breakdowns below falling HMA21 alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5488_12h_donchian20_1w_hma21_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1w data for HMA21 trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA21 on 1w close: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(arr, period):
            if len(arr) < period:
                return np.full(len(arr), np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        close_series = pd.Series(df_1w['close'])
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        wma_half = close_series.rolling(window=half_n, min_periods=half_n).apply(
            lambda x: np.dot(x, np.arange(1, half_n+1)) / np.arange(1, half_n+1).sum(), raw=True
        ).values
        wma_full = close_series.rolling(window=21, min_periods=21).apply(
            lambda x: np.dot(x, np.arange(1, 22)) / np.arange(1, 22).sum(), raw=True
        ).values
        
        # 2*WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full
        # WMA of raw_hma with period sqrt(n)
        hma_21 = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).apply(
            lambda x: np.dot(x, np.arange(1, sqrt_n+1)) / np.arange(1, sqrt_n+1).sum(), raw=True
        ).values
        
        # Align to LTF (12h) with shift(1) for completed bars only
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
        # HMA trend: price above HMA21 = bullish, below = bearish
        price_above_hma = close > hma_21_aligned
        price_below_hma = close < hma_21_aligned
    else:
        hma_21_aligned = np.full(n, np.nan)
        price_above_hma = np.full(n, False)
        price_below_hma = np.full(n, False)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 12h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 20, 14, 21)  # Donchian, volume avg, ATR warmup, 1w HMA21 lookback
    
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
            np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.0 * ATR below highest since entry
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price moves below HMA21 (trend weakening)
                if price <= stop_price or price <= donchian_low[i] or price < hma_21_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.0 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price moves above HMA21 (trend weakening)
                if price >= stop_price or price >= donchian_high[i] or price > hma_21_aligned[i]:
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
        
        # Entry conditions
        if breakout_up and volume_confirmed and price_above_hma[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and price_below_hma[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals