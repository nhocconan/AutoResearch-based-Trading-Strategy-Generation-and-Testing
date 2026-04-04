#!/usr/bin/env python3
"""
Experiment #5313: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: On 4h timeframe, price breaking above/below the 20-period Donchian channel 
with volume > 1.3x average and aligned with 12h Hull Moving Average (HMA-21) trend 
captures strong momentum moves while avoiding whipsaws. Long when breaking above 
Donchian upper with volume confirmation and HMA-21 trending up, short when breaking 
below Donchian lower with volume confirmation and HMA-21 trending down. Uses discrete 
position sizing (0.25) and ATR-based stoploss to control drawdown. Target: 19-50 
trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag while 
maintaining statistical significance. Works in both bull (breakouts continue) and 
bear (breakdowns continue) markets via symmetric long/short logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5313_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 12h data for HMA-21 trend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        # Hull Moving Average (HMA-21)
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='same')
        
        close_12h = df_12h['close'].values
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        raw_hma = 2 * wma_half - wma_full
        hma_21 = wma(raw_hma, sqrt_len)
        
        # Align to LTF (4h) with shift(1) for completed bars only
        hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
        # HMA trend: rising if current > previous, falling if current < previous
        hma_trend = np.diff(hma_21_aligned, prepend=hma_21_aligned[0])
        hma_trend_aligned = align_htf_to_ltf(prices, df_12h, hma_trend)
    else:
        hma_trend_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
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
    
    warmup = max(20, 20, 14, 21)  # Donchian, volume avg, ATR, HMA warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Trade during active sessions only ---
        hour = hours[i]
        # Avoid low liquidity: 21-23 UTC (between US close and Asia open)
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(hma_trend_aligned[i])):
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
                # 3. HMA trend turns down (momentum loss)
                if price <= stop_price or price <= donchian_low[i] or hma_trend_aligned[i] < 0:
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
                # 3. HMA trend turns up (momentum loss)
                if price >= stop_price or price >= donchian_high[i] or hma_trend_aligned[i] > 0:
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
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirmed = volume_ratio[i] > 1.3
        
        # HMA trend conditions
        hma_up = hma_trend_aligned[i-1] > 0   # Trending up
        hma_down = hma_trend_aligned[i-1] < 0 # Trending down
        
        # Entry conditions
        if breakout_up and volume_confirmed and hma_up:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and hma_down:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals