#!/usr/bin/env python3
"""
Experiment #012: 12h Donchian Breakout + 1d Trend + Volume + Choppiness Regime

HYPOTHESIS: 12h Donchian(20) breakout captures medium-term trend shifts with less
noise than 4h and fewer fakeouts than daily. Combined with 1d SMA200 for trend
direction and Choppiness Index to stay out of range-bound markets, this is a
proven pattern that worked on SOL (test Sharpe 1.10-1.38) and ETH (test Sharpe 1.47).

WHY 12h: Balances signal quality vs trade frequency. 4h overtrades (target 75-150
vs 400+ failures), 1d undertrades (30 trades = too few). 12h = 12-37 trades/year.

WHY IT WORKS BOTH MARKETS:
- Bull: Breakout above Donchian upper + price > SMA200 + volume spike = strong long
- Bear: Breakout below Donchian lower + price < SMA200 + volume spike = strong short
- Range (CHOP > 61.8): No entries, wait for breakout

TARGET: 50-150 total trades over 4 years (12-37/year). HARD MAX: 200.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_sma200_vol_chop_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel: upper = highest high, lower = lowest low"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, middle, lower

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range market (avoid)
    CHOP < 38.2 = trending market (trade)
    """
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
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
    
    # 1d SMA200 for trend direction (must call ONCE, aligned to 12h)
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Local 12h indicators ===
    donch_upper, donch_mid, donch_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume confirmation
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
    
    warmup = 220  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME FILTER (Choppiness) ===
        # Only trade in trending markets (CHOP < 38.2)
        is_trending = chop[i] < 38.2
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        price_below_1d_sma = close[i] < sma_1d_aligned[i]
        
        # === DONCHIAN SIGNALS ===
        upper = donch_upper[i]
        lower = donch_lower[i]
        mid = donch_mid[i]
        
        # Breakout: close above upper band (bullish) or below lower band (bearish)
        bullish_breakout = close[i] > upper
        bearish_breakout = close[i] < lower
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Only enter in trending regime
            if is_trending:
                # Long: Bullish breakout + price above SMA200 + volume spike
                if bullish_breakout and price_above_1d_sma and vol_spike:
                    desired_signal = SIZE
                
                # Short: Bearish breakout + price below SMA200 + volume spike
                if bearish_breakout and price_below_1d_sma and vol_spike:
                    desired_signal = -SIZE
        else:
            # === EXIT LOGIC ===
            # Exit if price crosses middle band
            if position_side > 0 and close[i] < mid:
                desired_signal = 0.0
            
            if position_side < 0 and close[i] > mid:
                desired_signal = 0.0
            
            # Time-based exit: hold at least 4 bars (2 days)
            bars_held = i - entry_bar
            if bars_held >= 4:
                # Exit if trend reverses (crosses SMA200)
                if position_side > 0 and price_below_1d_sma:
                    desired_signal = 0.0
                if position_side < 0 and price_above_1d_sma:
                    desired_signal = 0.0
        
        # === TRAILING STOPLOSS (2.0 ATR) ===
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