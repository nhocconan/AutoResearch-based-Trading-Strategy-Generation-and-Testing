#!/usr/bin/env python3
"""
Experiment #028: 4h Williams %R Extreme + Donchian Breakout + Volume Confirmation + 1d SMA50

HYPOTHESIS: Williams %R at extreme levels (-80/+20) identifies exhaustion points where
reversals are most likely. Combined with Donchian breakout for structure, volume for
institutional confirmation, and 1d SMA50 for trend alignment, this captures high-probability
reversals while filtering noise.

WHY IT WORKS IN BULL AND BEAR:
- Long: Williams %R < -80 (oversold) + breakout above Donchian high + price > SMA50
  = Bull market reversals with trend confirmation
- Short: Williams %R > +20 (overbought) + breakdown below Donchian low + price < SMA50
  = Bear market rallies that fail

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
Signal size: 0.30 (discrete levels).

KEY INSIGHT: Tighter extreme filters = fewer but higher-quality trades.
Williams %R < -80 is rare - only when truly oversold.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_williams_r_donchian_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - measures current price relative to high-low range"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 0:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

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
    willr = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (18 periods = 3 days on 4h - tighter for more signals)
    donchian_period = 18
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume (20-period MA)
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
    
    warmup = 100  # Need enough for indicators + Donchian(18)
    
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
        
        if np.isnan(willr[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === WILLIAMS %R EXTREME LEVELS ===
        # Long: Williams %R below -80 (strongly oversold) = reversal candidate
        # Short: Williams %R above -20 (overbought) = reversal candidate
        willr_oversold = willr[i] < -80
        willr_overbought = willr[i] > -20
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # === VOLUME CONFIRMATION ===
        # Require 1.8x average volume for confirmation
        vol_spike = vol_ratio[i] > 1.8
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Oversold + Breakout above Donchian high + Price > SMA50 ===
            # TIGHT: All three conditions must be met
            breakout_up = high[i] > prev_donchian_high
            
            if willr_oversold and breakout_up and price_above_1d_sma and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Overbought + Breakdown below Donchian low + Price < SMA50 ===
            breakout_down = low[i] < prev_donchian_low
            
            if willr_overbought and breakout_down and not price_above_1d_sma and vol_spike:
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
        
        # === TIME-BASED EXIT (hold at least 4 bars = 16 hours) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit if price crosses 20-period SMA (local mean reversion)
            sma_20_local = pd.Series(close[i - 19:i + 1]).mean() if i >= 19 else close[i]
            
            if position_side > 0 and close[i] < sma_20_local:
                desired_signal = 0.0
            if position_side < 0 and close[i] > sma_20_local:
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