#!/usr/bin/env python3
"""
Experiment #028: 6h Donchian Breakout/Pullback + 1d EMA200 Trend + Volume Regime

HYPOTHESIS: Combining breakout AND pullback entries within a Donchian channel
framework captures both momentum moves and mean reversion. The 1d EMA200 filters
for major trend direction, reducing whipsaws in counter-trend moves.

WHY 6h: Slower than 4h reduces fee drag by ~33%. 6h = 4 bars/day, captures
medium-term institutional moves without overtrading.

WHY IT WORKS IN BULL AND BEAR:
- Long: Price above 1d EMA200 + breakout above Donchian high OR pullback to mid-channel
- Short: Price below 1d EMA200 + breakdown below Donchian low OR rally to mid-channel
- Choppiness filter keeps us out of range-bound whipsaws
- Volume confirms institutional participation

TARGET: 50-150 total over 4 years = 12-37/year. HARD MAX: 300.
Signal size: 0.25 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_dual_entry_1d_ema200_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for major trend (more signals than SMA200)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (24 periods = 6 days on 6h)
    donchian_period = 24
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume indicators
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = max(200, donchian_period + 20)  # Need EMA200 + Donchian(24) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA200) ===
        is_bull_trend = close[i] > ema_1d_aligned[i]
        is_bear_trend = close[i] < ema_1d_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Skip if too choppy (CHOP > 61.8)
        is_choppy = chop[i] > 61.8
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.2
        
        # Current values
        current_high = high[i]
        current_low = low[i]
        
        # Previous bar values for breakout detection
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        prev_close = close[i - 1] if i > 0 else close[i]
        prev_donchian_mid = donchian_mid[i - 1] if i > 0 else 0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Skip if market is too choppy
            if is_choppy:
                signals[i] = 0.0
                continue
            
            # === LONG ENTRY 1: Breakout above Donchian high ===
            # Price breaks above previous 24-bar high with trend and volume
            if current_high > prev_donchian_high and is_bull_trend:
                if vol_confirm:
                    desired_signal = SIZE
            
            # === LONG ENTRY 2: Pullback to mid-channel in bull trend ===
            # Price retraces to Donchian mid after being near highs, with volume
            if is_bull_trend and not is_choppy:
                dist_to_mid = (close[i] - prev_donchian_low) / (prev_donchian_high - prev_donchian_low + 1e-10)
                # Near mid-channel (40-55% of range)
                if 0.40 <= dist_to_mid <= 0.55:
                    if vol_confirm:
                        desired_signal = SIZE
            
            # === SHORT ENTRY 1: Breakdown below Donchian low ===
            if current_low < prev_donchian_low and is_bear_trend:
                if vol_confirm:
                    desired_signal = -SIZE
            
            # === SHORT ENTRY 2: Rally to mid-channel in bear trend ===
            if is_bear_trend and not is_choppy:
                dist_to_mid = (close[i] - prev_donchian_low) / (prev_donchian_high - prev_donchian_low + 1e-10)
                # Near mid-channel (45-60% of range)
                if 0.45 <= dist_to_mid <= 0.60:
                    if vol_confirm:
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
        
        # === EXIT: Price crosses Donchian mid ===
        if in_position and desired_signal == 0.0:
            if position_side > 0 and close[i] < donchian_mid[i]:
                pass  # Keep signal at 0
            if position_side < 0 and close[i] > donchian_mid[i]:
                pass  # Keep signal at 0
        
        # === HOLDING PERIOD (minimum 8 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit if price crosses mid-channel
            if position_side > 0 and close[i] < donchian_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
                desired_signal = 0.0
        
        # === 4R TARGET (take profit at 4R) ===
        if in_position and position_side > 0:
            profit = close[i] - entry_price
            if profit >= 4.0 * entry_atr:
                desired_signal = SIZE / 2  # Reduce to half position
        
        if in_position and position_side < 0:
            profit = entry_price - close[i]
            if profit >= 4.0 * entry_atr:
                desired_signal = -SIZE / 2  # Reduce to half position
        
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