#!/usr/bin/env python3
"""
Experiment #5459: 6h Donchian(20) breakout + 12h Camarilla pivot regime + volume confirmation
HYPOTHESIS: On 6h timeframe, price breaking above/below the 20-period Donchian channel with 
volume > 2.0x average and aligned with the 12h Camarilla pivot regime (price between H3/L3 for continuation, 
or breaking R4/S4 for acceleration) captures strong momentum moves with multi-timeframe structure confirmation. 
Discrete position sizing (0.25) and ATR-based stoploss (2.0x ATR) control risk. Target: 12-37 trades/year 
(50-150 total over 4 years) to minimize fee drag while maintaining statistical significance. 
Works in bull markets via breakouts above rising Camarilla levels and in bear markets via short 
breakdowns below falling levels, with volume filtering false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5459_6h_donchian20_12h_camarilla_vol_v1"
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
    
    # === HTF: 12h data for Camarilla pivot levels ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 1:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # Calculate Camarilla pivot levels for 12h
        # Pivot = (H + L + C) / 3
        pivot_12h = (high_12h + low_12h + close_12h) / 3.0
        # Range = H - L
        range_12h = high_12h - low_12h
        
        # Camarilla levels:
        # H4 = pivot + (range * 1.1/2)
        # H3 = pivot + (range * 1.1/4)
        # L3 = pivot - (range * 1.1/4)
        # L4 = pivot - (range * 1.1/2)
        h4_12h = pivot_12h + (range_12h * 1.1 / 2)
        h3_12h = pivot_12h + (range_12h * 1.1 / 4)
        l3_12h = pivot_12h - (range_12h * 1.1 / 4)
        l4_12h = pivot_12h - (range_12h * 1.1 / 2)
        
        # Align to LTF (6h) with shift(1) for completed bars only
        h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h) if len(h4_12h) > 0 else np.full(n, np.nan)
        h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h) if len(h3_12h) > 0 else np.full(n, np.nan)
        l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h) if len(l3_12h) > 0 else np.full(n, np.nan)
        l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h) if len(l4_12h) > 0 else np.full(n, np.nan)
    else:
        h4_12h_aligned = np.full(n, np.nan)
        h3_12h_aligned = np.full(n, np.nan)
        l3_12h_aligned = np.full(n, np.nan)
        l4_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 20, 14)  # Donchian, volume avg, ATR warmup
    
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
            np.isnan(h4_12h_aligned[i]) or np.isnan(h3_12h_aligned[i]) or 
            np.isnan(l3_12h_aligned[i]) or np.isnan(l4_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Camarilla regime classification ---
        # Regime 1: Acceleration breakout (price > H4 or < L4)
        # Regime 2: Continuation zone (price between H3 and L3)
        # Regime 3: Extension zone (price between H4-H3 or L3-L4)
        regime_acceleration_up = price > h4_12h_aligned[i]
        regime_acceleration_down = price < l4_12h_aligned[i]
        regime_continuation = (price > h3_12h_aligned[i]) & (price < l3_12h_aligned[i])
        regime_extension_up = (price > h3_12h_aligned[i]) & (price < h4_12h_aligned[i])
        regime_extension_down = (price > l4_12h_aligned[i]) & (price < l3_12h_aligned[i])
        
        # --- Exit Logic: Close position on stoploss or regime failure ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.0 * ATR below highest since entry
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price moves into acceleration down regime (potential reversal)
                if price <= stop_price or price <= donchian_low[i] or regime_acceleration_down:
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
                # 3. Price moves into acceleration up regime (potential reversal)
                if price >= stop_price or price >= donchian_high[i] or regime_acceleration_up:
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
        
        # Volume confirmation: current volume > 2.0x average volume (stricter than 1.8x)
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Entry conditions: breakout with volume confirmation in favorable regime
        # Long: breakout up in continuation, extension up, or acceleration up regime
        # Short: breakout down in continuation, extension down, or acceleration down regime
        long_regime_favorable = regime_continuation | regime_extension_up | regime_acceleration_up
        short_regime_favorable = regime_continuation | regime_extension_down | regime_acceleration_down
        
        if breakout_up and volume_confirmed and long_regime_favorable:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and short_regime_favorable:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals