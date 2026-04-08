#!/usr/bin/env python3
"""
Experiment #4299: 6h Donchian(20) breakout + 12h EMA50 trend + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture swing momentum when aligned with 12h EMA50 trend (price > EMA50 for longs, < EMA50 for shorts) and confirmed by volume (>2.0x average). Uses 12h EMA for smoother trend filter than shorter periods while targeting 50-150 total trades over 4 years (12-37/year). ATR-based trailing stop (2.5x) for risk management. Position size 0.25 balances return and drawdown. Works in bull via breakout continuation, in bear via shorting breakdowns. Novelty: Uses 6h primary timeframe with 12h HTF as specified in experiment #4299 to balance noise reduction and trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4299_6h_donchian20_12h_ema_vol_v1"
timeframe = "6h"
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
    
    # === Precompute HTF: 12h EMA50 for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 50:
        ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    else:
        ema_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50)  # Donchian, vol MA, ATR, 12h EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # 12h EMA50 trend filter
            price_above_ema = price > ema_12h_aligned[i]
            price_below_ema = price < ema_12h_aligned[i]
            
            # Long conditions: Donchian breakout up + price above EMA50
            long_entry = breakout_up and price_above_ema
            
            # Short conditions: Donchian breakout down + price below EMA50
            short_entry = breakout_dn and price_below_ema
            
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
        else:
            signals[i] = 0.0
    
    return signals