#!/usr/bin/env python3
"""
Experiment #5844: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts aligned with weekly HMA trend capture strong momentum. 
Weekly HMA acts as smooth trend filter: breakouts above weekly HMA (uptrend) or below 
weekly HMA (downtrend) with volume confirmation indicate institutional participation. 
In ranging markets (price near weekly HMA), breakouts are filtered out, reducing false signals. 
Works in bull markets (breakouts above weekly HMA with volume) and bear markets 
(breakdowns below weekly HMA with volume). Targets 30-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5844_1d_donchian20_1w_hma_vol_v1"
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
    
    # === HTF: 1w data for HMA trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA(21) on weekly close
        # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        close_1w = df_1w['close'].values.astype(np.float64)
        if len(close_1w) >= 21:
            wma_half = wma(close_1w, half_len)
            wma_full = wma(close_1w, 21)
            hma_2xminus = 2 * wma_half - wma_full
            # Pad beginning with NaN
            hma_2xminus_padded = np.full(len(close_1w), np.nan)
            hma_2xminus_padded[half_len-1:len(wma_2xminus)+half_len-1] = hma_2xminus
            hma_1w = wma(hma_2xminus_padded[~np.isnan(hma_2xminus_padded)], sqrt_len)
            # Pad beginning with NaN
            hma_1w_full = np.full(len(close_1w), np.nan)
            start_idx = len(close_1w) - len(hma_1w)
            hma_1w_full[start_idx:] = hma_1w
        else:
            hma_1w_full = np.full(len(close_1w), np.nan)
        
        # Align to 1d timeframe
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_full)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 1d Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 20, 21, 14)  # Donchian, volume avg, weekly HMA, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below weekly HMA (trend change)
                if price <= stop_price or price <= hma_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above weekly HMA (trend change)
                if price >= stop_price or price >= hma_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        # Weekly HMA filter: breakout in direction of trend
        hma_long = price > hma_1w_aligned[i]
        hma_short = price < hma_1w_aligned[i]
        
        # Entry conditions: breakout in direction of weekly HMA with volume confirmation
        long_setup = breakout_up and hma_long and volume_confirmed
        short_setup = breakout_down and hma_short and volume_confirmed
        
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