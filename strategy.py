#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian Breakout + 1w SMA20 Trend + Volume

HYPOTHESIS: Daily Donchian(20) breakout captures institutional moves at key
structural points. Combined with 1w SMA20 as trend filter (prevents buying
breakouts in downtrends, shorting in uptrends) and volume confirmation, this
mirrors the proven 4h winning pattern but on 1d for fewer, higher-quality trades.

WHY 1d: 
- Fewer trades than 4h = less fee drag
- More significant structure = better signal quality  
- 1w SMA20 protects against bear markets (never buy in downtrend)
- Works in both bull (trend following) and bear (breakout shorting)

TARGET: 75-150 total trades over 4 years = 19-37/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA20 for trend direction
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=20, min_periods=20).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Local 1d indicators
    donchian_period = 20
    highest_20 = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_20 = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    warmup = 100  # Need 20 for Donchian + 20 for vol MA + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(highest_20[i]) or np.isnan(lowest_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1w SMA20) ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout: close above 20-day high = bullish breakout
        # Breakout: close below 20-day low = bearish breakout
        bull_breakout = close[i] > highest_20[i-1] if i > 0 else False
        bear_breakout = close[i] < lowest_20[i-1] if i > 0 else False
        
        # Previous bar also broke (confirms strength)
        bull_breakout_prev = close[i-1] > highest_20[i-2] if i > 1 else False
        bear_breakout_prev = close[i-1] < lowest_20[i-2] if i > 1 else False
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # Long: Bull breakout + price above 1w SMA + volume spike
            if bull_breakout and price_above_1w_sma and vol_spike:
                desired_signal = SIZE
            
            # Short: Bear breakout + price below 1w SMA + volume spike
            if bear_breakout and not price_above_1w_sma and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MINIMUM HOLD (3 bars = 3 days to avoid whipsaw) ===
        bars_held = i - entry_bar
        min_hold_bars = 3
        
        if in_position and bars_held < min_hold_bars:
            # Don't exit early
            desired_signal = position_side * SIZE
        
        # === TREND REVERSAL EXIT ===
        if in_position and bars_held >= min_hold_bars:
            # Exit long if price falls below 1w SMA (trend changed)
            if position_side > 0 and not price_above_1w_sma:
                desired_signal = 0.0
            
            # Exit short if price rises above 1w SMA (trend changed)
            if position_side < 0 and price_above_1w_sma:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
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
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals