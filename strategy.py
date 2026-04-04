#!/usr/bin/env python3
"""
Experiment #6398: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: 1d Donchian breakouts aligned with weekly HMA trend capture strong momentum while avoiding counter-trend trades. Weekly timeframe provides structural bias: only trade breakouts in direction of weekly HMA(21). Volume confirmation (>1.5x 20-bar average) ensures institutional participation. Discrete sizing (0.25) balances profit and drawdown. ATR trailing stop (2.0x) manages risk. Works in bull via trend-aligned breakouts, in bear via short breakdowns with volume confirmation and trend alignment. Target: 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6398_1d_donchian20_1w_hma_vol_v1"
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
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        def wma(arr, window):
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        hma_1w = 2 * wma_half - wma_full
        hma_1w = wma(hma_1w, sqrt_len)
        # Align to 1d timeframe
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume confirmation (current 1d volume vs 20-bar average) ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    warmup = max(20, 20, 14) + 1  # Donchian, volume avg, ATR lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(atr[i]) or
            np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]  # Volume filter: 1.5x 20-bar average
        
        # Trend filter: only trade in direction of weekly HMA
        # Long: price above weekly HMA
        # Short: price below weekly HMA
        trend_filter_up = price > hma_1w_aligned[i]
        trend_filter_down = price < hma_1w_aligned[i]
        
        # Entry logic:
        # Long: breakout up + volume confirmation + trend filter up
        # Short: breakout down + volume confirmation + trend filter down
        
        long_entry = breakout_up and volume_confirmed and trend_filter_up
        short_entry = breakout_down and volume_confirmed and trend_filter_down
        
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