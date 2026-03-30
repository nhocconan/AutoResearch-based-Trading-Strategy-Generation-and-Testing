#!/usr/bin/env python3
"""
Experiment #009: 4h Donchian(10) + 1d ADX + Vol Spike

HYPOTHESIS: 4h Donchian(10) captures medium-term momentum swings. 
1d ADX > 25 ensures trending conditions (no range-bound whipsaws).
Volume spike confirms institutional entry. ATR stoploss manages risk.
1d SMA confirms trend direction.

WHY IT WORKS IN BULL AND BEAR: Breakouts occur in both directions.
Long breakouts in uptrends, shorts in downtrends. ADX filter removes
range-bound chop that destroys trend followers in bear markets.

TARGET: 40-80 total trades over 4 years (10-20/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_adx_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """ADX indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with EMA
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DX
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if atr_smooth[i] > 1e-10:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 1e-10:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = EMA of DX
    adx = pd.Series(dx).ewm(span=period, min_periods=2*period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX for regime detection (trend vs chop)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d SMA50 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (10 periods = 40 hours = ~1.7 days)
    donchian_period = 10
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(adx_1d_aligned[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME FILTER: ADX > 25 = trending, ADX < 20 = choppy ===
        adx_trending = adx_1d_aligned[i] > 25
        adx_chop = adx_1d_aligned[i] < 20
        
        # === TREND DIRECTION: price vs 1d SMA50 ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # Volume confirmation (1.5x average)
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian breakout signals
        upper_break = close[i] > donchian_high[i] if not np.isnan(donchian_high[i]) else False
        lower_break = close[i] < donchian_low[i] if not np.isnan(donchian_low[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Upper Donchian breakout + trend up + volume + trending regime ===
            if upper_break and price_above_1d_sma and vol_spike and adx_trending:
                desired_signal = SIZE
            
            # === SHORT: Lower Donchian breakout + trend down + volume + trending regime ===
            if lower_break and not price_above_1d_sma and vol_spike and adx_trending:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD (6 bars = 1 day to reduce churn) ===
        bars_held = i - entry_bar
        if in_position and bars_held < 6:
            # Don't exit early due to opposite signal
            if position_side > 0 and desired_signal < 0:
                desired_signal = SIZE  # Keep long
            if position_side < 0 and desired_signal > 0:
                desired_signal = -SIZE  # Keep short
        
        # === TAKE PROFIT (3:1 R:R when ADX drops = trend weakening) ===
        if in_position and bars_held >= 6:
            if position_side > 0:
                profit_pct = (close[i] - entry_price) / entry_price
                if profit_pct > 0.05 and adx_chop:  # >5% gain and ADX drops = exit
                    desired_signal = 0.0
            if position_side < 0:
                profit_pct = (entry_price - close[i]) / entry_price
                if profit_pct > 0.05 and adx_chop:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals