#!/usr/bin/env python3
"""
Experiment #5547: 6h Donchian(20) breakout + 1d/1w pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 1.5x average and aligned with 
both 1d Camarilla pivot bias (price above/below pivot) and 1w trend (price above/below 200 EMA) 
capture high-probability continuation moves. The multi-timeframe alignment filters false breakouts 
while allowing participation in strong trends. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5547_6h_donchian20_1d1w_pivot_vol_v1"
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
    
    # === HTF: 1d data for Camarilla pivot bias ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate Camarilla pivot levels from previous day
        pivot = (df_1d['high'].shift(1) + df_1d['low'].shift(1) + df_1d['close'].shift(1)) / 3
        range_ = df_1d['high'].shift(1) - df_1d['low'].shift(1)
        r3 = pivot + range_ * 1.1 / 2
        s3 = pivot - range_ * 1.1 / 2
        # Bias: 1 = bullish (close above pivot), -1 = bearish (close below pivot)
        bias_1d = np.where(df_1d['close'].values > pivot.values, 1, -1)
        # Align to LTF (6h) with shift(1) for completed bars only
        bias_1d_aligned = align_htf_to_ltf(prices, df_1d, bias_1d)
    else:
        bias_1d_aligned = np.full(n, 0)  # neutral if insufficient data
    
    # === HTF: 1w data for 200 EMA trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 200:
        # Calculate EMA(200) on weekly data
        ema_200w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False).mean().values
        # Trend: 1 = uptrend (close above EMA200), -1 = downtrend (close below EMA200)
        trend_1w = np.where(df_1w['close'].values > ema_200w, 1, -1)
        # Align to LTF (6h) with shift(1) for completed bars only
        trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    else:
        trend_1w_aligned = np.full(n, 0)  # neutral if insufficient data
    
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
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 200, 14)  # Donchian, volume avg, weekly EMA warmup, ATR warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            i >= len(bias_1d_aligned) or i >= len(trend_1w_aligned)):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR Donchian lower band break OR bias/trend reversal
                if price <= stop_price or price <= donchian_low[i] or \
                   (bias_1d_aligned[i] == -1) or (trend_1w_aligned[i] == -1):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR Donchian upper band break OR bias/trend reversal
                if price >= stop_price or price >= donchian_high[i] or \
                   (bias_1d_aligned[i] == 1) or (trend_1w_aligned[i] == 1):
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
        
        # Only trade when 1d bias and 1w trend agree
        long_entry = breakout_up and volume_confirmed and \
                   (bias_1d_aligned[i] == 1) and (trend_1w_aligned[i] == 1)
        short_entry = breakout_down and volume_confirmed and \
                   (bias_1d_aligned[i] == -1) and (trend_1w_aligned[i] == -1)
        
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