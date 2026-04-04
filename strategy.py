#!/usr/bin/env python3
"""
Experiment #5054: 1h Donchian(20) Breakout + 4h EMA50 Trend + 1d Volume Spike + Session Filter
HYPOTHESIS: On 1h timeframe, Donchian(20) breakouts aligned with 4h EMA50 trend and confirmed by 1d volume spikes capture strong momentum. Session filter (08-20 UTC) reduces noise from low-liquidity periods. Using 4h for trend direction and 1d for volume confirmation ensures trades align with higher timeframe structure while using 1h only for precise entry timing. Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag. Position size fixed at 0.20 to balance risk and reward.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5054_1h_donchian20_4h_ema50_1d_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 4h data for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    # Precompute HTF: 1d data for volume spike
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: EMA50 for trend direction ===
    if len(df_4h) >= 50:
        close_4h = df_4h['close'].values.astype(np.float64)
        ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Volume spike (2x average) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values.astype(np.float64)
        vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.ones(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_1d[20:]
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC ===
    # open_time is already datetime64[ns], access via index
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 50, 20, 14)  # Donchian, 4h EMA50, 1d volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        hour = hours[i]
        
        # Session filter: only trade 08-20 UTC
        if hour < 8 or hour > 20:
            if in_position:
                # Still manage exits during off-hours
                if position_side > 0:  # Long
                    highest_since_entry = max(highest_since_entry, high[i])
                    if price < highest_since_entry - 2.5 * atr[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:  # Short
                    lowest_since_entry = min(lowest_since_entry, low[i])
                    if price > lowest_since_entry + 2.5 * atr[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
            else:
                signals[i] = 0.0
            continue
        
        # --- Exit Logic (during session) ---
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
        
        # --- New Position Entry Logic (during session) ---
        # Trend filter: price above/below 4h EMA50
        uptrend = price > ema_50_aligned[i]
        downtrend = price < ema_50_aligned[i]
        
        # Volume filter: 1d volume > 2x average
        vol_confirm = vol_ratio_1d_aligned[i] > 2.0
        
        # Donchian breakout conditions
        breakout_long = (price >= high_roll[i]) and uptrend and vol_confirm
        breakout_short = (price <= low_roll[i]) and downtrend and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals