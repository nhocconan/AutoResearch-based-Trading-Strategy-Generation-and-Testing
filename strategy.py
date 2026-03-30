#!/usr/bin/env python3
"""
Experiment #022: 4h Donchian Breakout + 1d KAMA Trend + Volume Spike

HYPOTHESIS: Simple price-channel breakout strategy that works in both bull and bear:
- Donchian(20) 4h: Captures momentum breakouts from consolidation
- KAMA(10) 1d: Adaptive trend direction without SMA lag
- Volume spike: Confirms breakout legitimacy
- Choppiness Index: Avoid whipsaws in range-bound markets

WHY IT WORKS IN BOTH MARKETS:
- Bull: Price breaks above Donchian high + KAMA rising = strong long entry
- Bear: Price breaks below Donchian low + KAMA falling = strong short entry
- Range (CHOP > 61.8): No entries = avoids whipsaws in 2022 crash

KEY INSIGHT from DB: Best performers use simple conditions with 75-300 trades over 4 years.
This strategy targets 100-200 total trades with minimal fee drag.

TARGET: 100-200 total trades over 4 years (25-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_kama_vol_chop_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    - period: lookback for ER
    - fast/slow: EMA smoothing constants
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Price change
    change = np.abs(close[period:] - close[:-period])
    
    # Volatility (sum of absolute price changes)
    volatility = np.zeros(n)
    volatility[period] = np.sum(np.abs(close[1:period+1] - close[:period]))
    for i in range(period + 1, n):
        volatility[i] = volatility[i-1] - volatility[i-1] / period + abs(close[i] - close[i-1])
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    er[period:] = change / (volatility[period:] + 1e-10)
    
    # Smoothing constant
    er_sq = er ** 2
    fast_const = (2 / (fast + 1)) ** 2
    slow_const = (2 / (slow + 1)) ** 2
    sc = er_sq * (fast_const - slow_const) + slow_const
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period] = close[period]  # Start with SMA
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper/lower bands"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures trend vs range
    CHOP > 61.8 = choppy/range (avoid trading)
    CHOP < 38.2 = trending (good for momentum)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d KAMA for HTF trend ===
    kama_1d = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_prev = np.roll(kama_1d, 1)
    kama_1d_prev[0] = kama_1d[0]
    
    # HTF trend: KAMA rising = bull, falling = bear
    htf_trend_up = kama_1d > kama_1d_prev
    htf_trend_down = kama_1d < kama_1d_prev
    
    # Align HTF to 4h
    htf_up_aligned = align_htf_to_ltf(prices, df_1d, htf_trend_up.astype(float))
    htf_down_aligned = align_htf_to_ltf(prices, df_1d, htf_trend_down.astype(float))
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel (20 periods = 5 days of 4h bars)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
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
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Donchian needs 20, chop needs 14
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER ===
        # CHOP > 61.8 = range market, skip (no trades)
        # CHOP < 50 = trending, trade
        is_trending = chop[i] < 55.0  # Slightly relaxed for more trades
        
        # === BREAKOUT SIGNALS ===
        # Price breaks above 20-bar high = bullish breakout
        bullish_breakout = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
        # Price breaks below 20-bar low = bearish breakout
        bearish_breakout = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.5
        
        # === HTF TREND ===
        htf_up = htf_up_aligned[i] > 0.5 if not np.isnan(htf_up_aligned[i]) else False
        htf_down = htf_down_aligned[i] > 0.5 if not np.isnan(htf_down_aligned[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume + trending market + (HTF bull OR neutral)
            if bullish_breakout and vol_confirm and is_trending:
                # For long: prefer HTF bull OR neutral (down only when strong bear)
                if htf_up or not htf_down:  # Bull or neutral HTF
                    desired_signal = SIZE
            
            # SHORT: Bearish breakout + volume + trending market + (HTF bear OR neutral)
            elif bearish_breakout and vol_confirm and is_trending:
                # For short: prefer HTF bear OR neutral (up only when strong bull)
                if htf_down or not htf_up:  # Bear or neutral HTF
                    desired_signal = -SIZE
        
        # === STOPLOSS & EXIT (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if breaks below Donchian lower (structure fails)
                if close[i] < donchian_lower[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_down and htf_down_aligned[i] > 0.5:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if breaks above Donchian upper (structure fails)
                if close[i] > donchian_upper[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_up and htf_up_aligned[i] > 0.5:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
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