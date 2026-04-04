#!/usr/bin/env python3
"""
Experiment #5575: 6h Elder Ray Index + 1w Regime Filter + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Elder Ray Bull/Bear Power (EMA13-based) filtered by weekly trend (price > weekly EMA50 = bull regime, < = bear regime) with volume > 2.0x average captures high-probability momentum moves. In bull regime, long when Bull Power > 0 and rising; in bear regime, short when Bear Power < 0 and falling. Weekly regime prevents counter-trend trading, reducing whipsaw in sideways markets. ATR-based stoploss limits drawdown. Target: 12-37 trades/year (50-150 total over 4 years) with discrete position sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5575_6h_elder_ray_1w_regime_vol_v1"
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
    
    # === HTF: 1w data for regime filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        weekly_ema = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
        weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
        bull_regime = close > weekly_ema_aligned  # price above weekly EMA50 = bull regime
        bear_regime = close < weekly_ema_aligned  # price below weekly EMA50 = bear regime
    else:
        weekly_ema_aligned = np.full(n, np.nan)
        bull_regime = np.zeros(n, dtype=bool)
        bear_regime = np.zeros(n, dtype=bool)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray Bull/Bear Power ===
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
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
    
    warmup = max(50, 13, 20, 14)  # weekly EMA, EMA13, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR Elder Ray turns negative
                if price <= stop_price or bull_power[i] <= 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR Elder Ray turns positive
                if price >= stop_price or bear_power[i] >= 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Bull regime: long when Bull Power > 0 and rising (previous < current)
        # Bear regime: short when Bear Power < 0 and falling (previous > current)
        long_entry = bull_regime[i] and volume_confirmed and (bull_power[i] > 0) and (i > warmup and bull_power[i] > bull_power[i-1])
        short_entry = bear_regime[i] and volume_confirmed and (bear_power[i] < 0) and (i > warmup and bear_power[i] < bear_power[i-1])
        
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

</think>