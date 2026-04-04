#!/usr/bin/env python3
"""
Experiment #5334: 1h Donchian(20) breakout + 4h EMA(50) trend + 1d volume confirmation + session filter (08-20 UTC)
HYPOTHESIS: On 1h timeframe, price breaking above/below the 20-period Donchian channel with 
volume > 1.5x daily average and aligned with 4h EMA(50) trend captures strong momentum moves 
while minimizing false breakouts. Uses 1d HTF for volume filter to reduce noise. 
Session filter (08-20 UTC) avoids low liquidity periods. Discrete position sizing (0.20) 
and ATR-based stoploss control drawdown. Target: 15-37 trades/year (60-150 total over 4 years) 
on 1h timeframe to minimize fee drag while maintaining statistical significance. 
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5334_1h_donchian20_4h_ema_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 4h data for EMA50 trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 50:
        # Calculate EMA(50) on 4h close
        ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50 = np.array([])
    
    # Align to LTF (1h) with shift(1) for completed bars only
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50) if len(ema_50) > 0 else np.full(n, np.nan)
    
    # === HTF: 1d data for volume confirmation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        # Calculate average volume over 20 periods on 1d
        avg_volume_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    else:
        avg_volume_1d = np.array([])
    
    # Align to LTF (1h) with shift(1) for completed bars only
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d) if len(avg_volume_1d) > 0 else np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50)  # Donchian, volume avg, ATR, EMA50 warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Trade only during 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.0 * ATR below highest since entry
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price crosses below 4h EMA50 (trend reversal)
                if price <= stop_price or price <= donchian_low[i] or price < ema_50_aligned[i]:
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
                # 3. Price crosses above 4h EMA50 (trend reversal)
                if price >= stop_price or price >= donchian_high[i] or price > ema_50_aligned[i]:
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
        
        # Volume confirmation: current volume > 1.5x daily average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # 4h EMA50 trend filter
        # Long: price above 4h EMA50 (uptrend)
        # Short: price below 4h EMA50 (downtrend)
        ema_trend_up = price > ema_50_aligned[i-1]
        ema_trend_down = price < ema_50_aligned[i-1]
        
        # Entry conditions
        if breakout_up and volume_confirmed and ema_trend_up:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and ema_trend_down:
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