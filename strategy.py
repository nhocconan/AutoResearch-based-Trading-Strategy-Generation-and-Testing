#!/usr/bin/env python3
"""
Experiment #021 (REWRITE): Simple Donchian Breakout + Volume + Choppiness (4h)

HYPOTHESIS: Keep it brutally simple - one proven breakout signal with minimal filters.
The DB winners (Sharpe 1.3-1.5) all use simple Donchian/pivot + volume + regime.
Every additional condition multiplies the "AND" gate, collapsing trade count to zero.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Break 20-high with volume + chop = long. 2021 and 2024-2025 uptrends.
- Bear: Break 20-low with volume + chop = short. 2022 crash captured.
- Range: Choppiness filter prevents whipsaws in sideways markets.
- Simple structure = generalizes to unseen market conditions.

KEY INSIGHT from 27 failures: STRATEGY COMPLEXITY IS THE ENEMY.
The #1 reason for failure: too many stacked conditions = zero trades.
Solution: 2 conditions for entry (Donchian break + volume) + 1 filter (chop).

TARGET: 100-250 total trades over 4 years (25-62/year) - RELAXED from current zero trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_simple_v4"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout system"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, middle, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - identifies trending vs ranging markets
    CHOP > 61.8 = choppy/ranging (good for mean reversion)
    CHOP < 38.2 = trending (good for trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def calculate_ema(data, period, min_periods=None):
    """Exponential Moving Average"""
    if min_periods is None:
        min_periods = period
    return pd.Series(data).ewm(span=period, min_periods=min_periods, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF: EMA trend (simple and robust) ===
    htf_ema = calculate_ema(df_1d['close'].values, 21, min_periods=14)
    htf_ema_aligned = align_htf_to_ltf(prices, df_1d, htf_ema)
    
    # HTF trend direction
    htf_price = df_1d['close'].values
    htf_bullish = htf_price > htf_ema
    htf_bearish = htf_price < htf_ema
    
    htf_bullish_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish.astype(float))
    htf_bearish_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish.astype(float))
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    upper, middle, lower = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    # Trailing stop tracking
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Donchian needs 20, chop needs 14, volume needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === SIMPLE CONDITIONS ===
        
        # Donchian breakout (main signal)
        bullish_breakout = close[i] > upper[i] and close[i-1] <= upper[i-1]
        bearish_breakout = close[i] < lower[i] and close[i-1] >= lower[i-1]
        
        # Volume confirmation (relaxed: 1.5x instead of 1.8x)
        vol_spike = vol_ratio[i] > 1.5
        
        # Choppiness filter (relaxed: > 50 to get more trades)
        # High chop = ranging = breakout more likely to succeed
        choppy = chop[i] > 50.0
        
        # === HTF TREND (simple filter) ===
        htf_bull = htf_bullish_aligned[i] > 0.5 if not np.isnan(htf_bullish_aligned[i]) else True
        htf_bear = htf_bearish_aligned[i] > 0.5 if not np.isnan(htf_bearish_aligned[i]) else True
        
        # === ENTRY LOGIC (SIMPLIFIED) ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume + chop + HTF bull (or neutral)
            if bullish_breakout and vol_spike and choppy:
                if htf_bull or (not htf_bull and not htf_bear):  # Bull or neutral
                    desired_signal = SIZE
            
            # SHORT: Bearish breakout + volume + chop + HTF bear (or neutral)
            if bearish_breakout and vol_spike and choppy:
                if htf_bear or (not htf_bull and not htf_bear):  # Bear or neutral
                    desired_signal = -SIZE
        
        # === STOPLOSS and EXIT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR from highest point
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bearish_aligned[i] > 0.5:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR from lowest point
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bullish_aligned[i] > 0.5:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals