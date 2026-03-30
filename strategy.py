#!/usr/bin/env python3
"""
Experiment #028: 4h Donchian(15) + Volume + 1d SMA50 Trend + Simple ATR Stoploss

HYPOTHESIS: Simplest possible Donchian breakout should work better than complex strategies.
Previous attempts failed due to overcomplication (too many conditions = too many trades).
By using Donchian(15) (3-day channel for more signals) with just volume confirmation and
1d SMA50 trend filter, we capture institutional breakouts without fee drag.

KEY INSIGHT: DB top performers use Donchian + volume + ONE trend filter (HMA, KAMA, or SMA).
Adding more conditions (Williams R, Elder Ray, TRIX, etc.) causes overtrading or 0 trades.

WHY 4h: Middle ground - slower than 1h (less fee drag), faster than 12h (more opportunities).
Donchian(15) on 4h = 2.5-day channel - captures short-term breakouts within larger trends.

TARGET: 75-200 total trades over 4 years (19-50/year).
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian15_vol_1d_sma50_v1"
timeframe = "4h"
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
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (15 periods = 2.5 days on 4h - tighter for more signals)
    donchian_period = 15
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume moving average (20 periods = 3.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 80  # Need enough for Donchian(15) + volume MA(20) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        price_below_1d_sma = close[i] < sma_1d_aligned[i]
        
        # === DONCHIAN SIGNALS ===
        current_high = high[i]
        current_low = low[i]
        
        # Previous bar's Donchian values (for confirmed breakout)
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high with volume ===
            if current_high > prev_donchian_high and price_above_1d_sma and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low with volume ===
            if current_low < prev_donchian_low and price_below_1d_sma and vol_spike:
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
        
        # === MID-CHANNEL EXIT (mean reversion safety) ===
        if in_position and not stoploss_triggered:
            bars_held = i - entry_bar
            
            # Exit if price reverts to middle of channel after holding enough
            if bars_held >= 8:  # Hold at least 2 days on 4h
                if position_side > 0 and close[i] < donchian_mid[i]:
                    desired_signal = 0.0
                if position_side < 0 and close[i] > donchian_mid[i]:
                    desired_signal = 0.0
            
            # Take profit at 3R
            profit_target = SIZE * 3.0
            if position_side > 0:
                pnl_pct = (close[i] - entry_price) / entry_price
                if pnl_pct >= 0.075:  # ~7.5% move
                    desired_signal = SIZE / 2  # Half position
            if position_side < 0:
                pnl_pct = (entry_price - close[i]) / entry_price
                if pnl_pct >= 0.075:
                    desired_signal = -SIZE / 2
        
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