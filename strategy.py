#!/usr/bin/env python3
"""
Experiment #023: 12h Donchian(20) + Volume Spike + Choppiness Regime + 1d EMA Trend

HYPOTHESIS: Simple 3-condition system on 12h with proven regime filter captures
momentum breakouts while avoiding range-bound whipsaws.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- 12h timeframe balances trade frequency with signal quality
- Donchian(20) on 12h ≈ 10-day breakout window — captures medium-term moves
- Choppiness Index < 38.2 ensures we only trade in trending markets (proven filter)
- 1d EMA200 provides long-term direction bias
- Volume confirmation filters noise breakouts
- 4-bar minimum hold reduces fee churn from whipsaws

EXPECTED TRADE COUNT: 100-200 total over 4 years (25-50/year)
- 12h: 730 bars/year
- Donchian(20) breakouts: ~18-36/year potential
- Volume filter (1.5x): cuts 40% → 11-22/year
- Choppiness filter (<38.2): cuts 40% → 7-13/year
- 1d EMA200 bias: cuts 20% → 5-10/year
- Result: 20-40/year = 80-160 over 4 years ✓

PATTERN SOURCE: mtf_4h_chop_donchian_vol_regime_12h_v1 (test Sharpe 1.49, 107 trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_chop_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range - vectorized"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies trending vs ranging markets
    < 38.2 = trending, > 61.8 = ranging/choppy
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Vectorized ATR calculation
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling high-low range
    period_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    period_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    period_range = period_high - period_low
    
    # Choppiness = 100 * log10(sum_ATR) / log10(range)
    chop = np.full(n, np.nan)
    valid = (period_range > 0) & (~np.isnan(atr_sum))
    chop[valid] = 100 * np.log10(atr_sum[valid]) / np.log10(period_range[valid])
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF indicators (1d) ===
    # 1d EMA200 for long-term trend bias
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === Primary 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume average (20 bars = ~10 days at 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Choppiness Index (14 bars = 7 days at 12h)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 250  # Ensure all indicators ready
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema200_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # Volume spike confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Choppiness regime filter - only trade in trending markets
        is_trending = chop[i] < 38.2
        
        # 1d EMA200 trend bias
        price_above_ema1d = close[i] > ema200_1d_aligned[i]
        price_below_ema1d = close[i] < ema200_1d_aligned[i]
        
        if not in_position:
            # === LONG ENTRY: Breakout above Donchian high ===
            bullish_breakout = high[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
            
            if bullish_breakout and vol_spike and is_trending and price_above_ema1d:
                desired_signal = SIZE
                
            # === SHORT ENTRY: Breakdown below Donchian low ===
            bearish_breakout = low[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
            
            if bearish_breakout and vol_spike and is_trending and price_below_ema1d:
                desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR from highest point
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR from lowest point
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 4 bars (2 days at 12h) to avoid fee churn ===
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
        
        signals[i] = desired_signal
    
    return signals