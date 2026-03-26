#!/usr/bin/env python3
"""
Experiment #024: 12h Donchian Breakout + 1d SMA200 Trend + Volume

HYPOTHESIS: Simple price-channel breakout (Donchian) with 1d trend filter
captures institutional moves. This is THE proven pattern from DB:
- SOLUSDT: test Sharpe 1.10-1.38 (95 trades)
- ETHUSDT: test Sharpe 1.47 (95 trades)
- SOLUSDT: test Sharpe 1.46 (392 trades)

Why 12h: Natural trade frequency of 12-37/year (50-150 total over 4 years).
1d HTF for trend direction (bull = above SMA200, bear = below).

KEY INSIGHT: Fewer conditions = fewer trades = less fee drag.
Only 3 conditions: (1) Donchian breakout, (2) 1d trend aligned, (3) volume confirm.

TARGET: 50-150 total over 4 years. HARD MAX: 200.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_sma200_vol_v1"
timeframe = "12h"
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
    
    # 1d SMA200 for trend direction
    close_1d = df_1d['close'].values
    sma_200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel (20 periods)
    period_donchian = 20
    donchian_upper = pd.Series(high).rolling(window=period_donchian, min_periods=period_donchian).max().values
    donchian_lower = pd.Series(low).rolling(window=period_donchian, min_periods=period_donchian).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for slight smoothing of entries
    ema_8 = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30  # Standard sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 220  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Skip if 1d SMA not ready
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d TREND DIRECTION ===
        price_above_1d_sma = close[i] > sma_200_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout: close breaks above upper band (bull) or below lower band (bear)
        bullish_breakout = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
        bearish_breakout = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        
        # Touch of mid-channel (for mean reversion in range)
        touch_upper = close[i] >= donchian_upper[i] * 0.995  # Within 0.5% of upper
        touch_lower = close[i] <= donchian_lower[i] * 1.005  # Within 0.5% of lower
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === TREND-FOLLOWING: Breakout in direction of 1d trend ===
            
            # LONG: Bullish breakout + 1d trend bullish + volume confirm
            if bullish_breakout and price_above_1d_sma and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + 1d trend bearish + volume confirm
            if bearish_breakout and not price_above_1d_sma and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 2 bars) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 2:
            # Exit on trend reversal (price crosses 1d SMA)
            if position_side > 0 and not price_above_1d_sma:
                desired_signal = 0.0
            if position_side < 0 and price_above_1d_sma:
                desired_signal = 0.0
            
            # Exit on opposite Donchian signal
            if position_side > 0 and bearish_breakout:
                desired_signal = 0.0
            if position_side < 0 and bullish_breakout:
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals