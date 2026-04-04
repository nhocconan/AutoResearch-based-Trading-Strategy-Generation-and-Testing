#!/usr/bin/env python3
"""
Experiment #6011: 6h Donchian(20) breakout + 1d Camarilla pivot levels + volume confirmation
HYPOTHESIS: Camarilla pivot levels from 1d provide institutional support/resistance zones. 
Breakouts beyond R4/S4 with volume confirmation capture strong momentum, while reversals at R3/S3 
offer mean-reversion opportunities. 6h timeframe balances trade frequency and signal quality.
Uses discrete position sizing (0.25) and ATR-based stops to manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6011_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate Camarilla levels from previous day's OHLC
        # R4 = Close + 1.5*(High-Low), R3 = Close + 1.1*(High-Low)
        # S3 = Close - 1.1*(High-Low), S4 = Close - 1.5*(High-Low)
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        
        # Avoid division by zero and handle first bar
        diff = prev_high - prev_low
        diff = np.where(diff == 0, 1e-10, diff)
        
        r4 = prev_close + 1.5 * diff
        r3 = prev_close + 1.1 * diff
        s3 = prev_close - 1.1 * diff
        s4 = prev_close - 1.5 * diff
        
        # Align to 6h timeframe with shift(1) for completed bars only
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        r4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 14) + 1  # Donchian, volume avg, ATR lookbacks + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                # OR reversal at R3 (take profit at resistance)
                if price <= stop_price or price <= donchian_low[i] or price >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                # OR reversal at S3 (take profit at support)
                if price >= stop_price or price >= donchian_high[i] or price <= s3_aligned[i]:
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
        
        # Camarilla-based entries:
        # Long: breakout above R4 with volume (strong bullish continuation)
        # Short: breakdown below S4 with volume (strong bearish continuation)
        long_breakout = breakout_up and volume_confirmed and price > r4_aligned[i-1]
        short_breakout = breakout_down and volume_confirmed and price < s4_aligned[i-1]
        
        # Mean reversion entries at extreme levels:
        # Long: reversal from S3 with volume (oversold bounce)
        # Short: reversal from R3 with volume (overbought rejection)
        long_reversion = (price <= s3_aligned[i] and 
                         close[i] > open[i] and  # bullish candle
                         volume_confirmed)
        short_reversion = (price >= r3_aligned[i] and 
                          close[i] < open[i] and  # bearish candle
                          volume_confirmed)
        
        long_setup = long_breakout or long_reversion
        short_setup = short_breakout or short_reversion
        
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