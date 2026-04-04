#!/usr/bin/env python3
"""
Experiment #6413: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts with volume confirmation (>1.8x avg) and 12h HMA(21) trend filter capture strong momentum moves while avoiding false breakouts in choppy markets. The 12h HMA acts as a higher-timeframe trend filter (bullish when price > HMA, bearish when price < HMA), allowing only trades in the direction of the 12h trend. This reduces whipsaws during ranging periods and improves win rate. Volume confirmation ensures breakouts have institutional participation. Discrete sizing (0.25) balances profit potential and drawdown control. Target: 75-200 trades over 4 years. Works in bull via upward breakouts with volume and uptrend HMA, and in bear via downward breakouts with volume and downtrend HMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6413_4h_donchian20_12h_hma_vol_v1"
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
    
    # === HTF: 12h data for HMA(21) trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        # Calculate HMA(21) on 12h close prices
        # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 12  # sqrt(21) ≈ 4.58, but using n/2=10.5 -> 12 for integer window
        sqrt_len = 4   # sqrt(21) ≈ 4.58 -> 4
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        close_12h = df_12h['close'].values
        wma_half = np.concatenate([np.full(half_len-1, np.nan), wma(close_12h, half_len)])
        wma_full = np.concatenate([np.full(21-1, np.nan), wma(close_12h, 21)])
        raw_hma = 2 * wma_half - wma_full
        hma_12h = np.concatenate([np.full(sqrt_len-1, np.nan), wma(raw_hma[~np.isnan(raw_hma)], sqrt_len)]) if np.sum(~np.isnan(raw_hma)) >= sqrt_len else np.full_like(raw_hma, np.nan)
        
        # Align to 4h timeframe (shifted by 1 bar for lookback safety)
        hma_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 14, 21) + 1  # Donchian, volume avg, ATR, HMA lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                # 3. Price crosses below 12h HMA (trend change)
                if price <= stop_price or price <= donchian_low[i] or price < hma_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                # 3. Price crosses above 12h HMA (trend change)
                if price >= stop_price or price >= donchian_high[i] or price > hma_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8  # Volume filter
        
        # Entry logic: Only trade in direction of 12h HMA trend
        # Long: breakout up + volume + price above 12h HMA (uptrend)
        # Short: breakout down + volume + price below 12h HMA (downtrend)
        long_entry = breakout_up and volume_confirmed and (price > hma_aligned[i])
        short_entry = breakout_down and volume_confirmed and (price < hma_aligned[i])
        
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