#!/usr/bin/env python3
"""
Experiment #5268: 12h Donchian Breakout + 1w HMA Trend + Volume Spike + ATR Stop
HYPOTHESIS: On 12h timeframe, price breaking above/below Donchian(20) channels with 1w HMA(21) trend confirmation and volume > 1.5x 20-period average captures strong momentum moves. The 1w HMA filter ensures we only trade with the higher timeframe trend, reducing whipsaws. Volume spike confirms institutional participation. ATR-based stoploss limits downside. Designed for 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize fee drag. Works in bull markets by catching breakouts and in bear markets by catching breakdowns, while avoiding ranging conditions via volume and trend filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5268_12h_donchian20_1w_hma_vol_v1"
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
    
    # === HTF: 1w data for HMA trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Hull Moving Average calculation
        def hull_moving_average(arr, period):
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))
            wma1 = pd.Series(arr).ewm(span=half_period, adjust=False).mean()
            wma2 = pd.Series(arr).ewm(span=period, adjust=False).mean()
            raw_hma = 2 * wma1 - wma2
            hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
            return hma.values
        
        hma_21 = hull_moving_average(df_1w['close'].values, 21)
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channels (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 12h Indicators: Volume Spike (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    vol_ratio = volume / vol_ma_20  # Current volume vs 20-period average
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
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
    
    warmup = max(20, 21, 14)  # Donchian, HMA, ATR warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 00-24 UTC (12h timeframe, full coverage) ---
        # 12h candles already cover major sessions
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on Donchian reversal or ATR stop ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_price = entry_price - 2.5 * atr[i]
                if price < stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Trail stop: update highest price
                highest_since_entry = max(highest_since_entry, price)
                # Exit if price drops 1.5*ATR from high
                if price < highest_since_entry - 1.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian breakdown (price < lower channel)
                if price < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                signals[i] = SIZE
            else:  # Short position
                stop_price = entry_price + 2.5 * atr[i]
                if price > stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Trail stop: update lowest price
                lowest_since_entry = min(lowest_since_entry, price)
                # Exit if price rises 1.5*ATR from low
                if price > lowest_since_entry + 1.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian breakout (price > upper channel)
                if price > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Trend filter from 1w HMA
        hma_trend_up = hma_21_aligned[i] > hma_21_aligned[i-1] if i > 0 else False
        hma_trend_down = hma_21_aligned[i] < hma_21_aligned[i-1] if i > 0 else False
        
        # Volume confirmation
        volume_spike = vol_ratio[i] > 1.5  # Volume > 1.5x 20-period average
        
        # Donchian breakout
        breakout_up = price > donchian_high[i]
        breakout_down = price < donchian_low[i]
        
        # Entry conditions: Trend + Volume + Breakout
        if hma_trend_up and volume_spike and breakout_up:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = price
            lowest_since_entry = price
            signals[i] = SIZE
        elif hma_trend_down and volume_spike and breakout_down:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = price
            lowest_since_entry = price
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>