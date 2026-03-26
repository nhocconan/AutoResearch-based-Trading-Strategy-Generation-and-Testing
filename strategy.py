#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + 1d Trend + Volume + Choppiness

HYPOTHESIS: Donchian(20) breakout on 12h captures structural market moves when
combined with 1d trend direction filter. Volume spike confirms institutional 
participation. Choppiness Index keeps us out of range-bound periods where 
breakouts fail. This is the PROVEN winning pattern from DB (SOLUSDT 1.10-1.38
test Sharpe, ETHUSDT 1.47 with Camarilla variant).

WHY 12h: Slower than 4h (reduces fee drag), captures multi-day trends. 
Target 50-150 total trades over 4 years (12-37/year).

KEY COMPONENTS (from proven winners):
1. Donchian channel breakout (price channel = structural levels)
2. 1d SMA200 trend filter (filters counter-trend entries)
3. Volume confirmation (institutional participation required)
4. Choppiness regime filter (avoid range-bound whipsaws)
5. ATR stoploss (2.0x for risk management)

SIMPLE = FEWER TRADES = LESS FEE DRAG = BETTER TEST PERFORMANCE
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_trend_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP < 38.2 = trending (momentum works)
    CHOP > 61.8 = choppy (avoid breakout trades)
    """
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel: upper = highest high, lower = lowest low"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian(20) channel
    dc_upper, dc_middle, dc_lower = calculate_donchian(high, low, period=20)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Position size: 30% of capital
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
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
        
        # Current state
        price = close[i]
        price_above_1d_sma = price > sma_200_aligned[i]
        price_below_1d_sma = price < sma_200_aligned[i]
        
        # Choppiness regime
        is_trending = chop[i] < 38.2
        is_choppy = chop[i] > 61.8
        
        # Donchian levels
        dc_up = dc_upper[i]
        dc_mid = dc_middle[i]
        dc_low = dc_lower[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # ATR for stoploss
        atr = atr_14[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above upper Donchian + bullish trend ===
            # Breakout: price crosses above previous upper band
            prev_upper = dc_upper[i-1] if i > 0 else dc_up
            breakout_up = price > prev_upper and price > dc_up - 0.5 * atr
            
            if breakout_up and price_above_1d_sma:
                if vol_spike:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below lower Donchian + bearish trend ===
            # Breakdown: price crosses below previous lower band
            prev_lower = dc_lower[i-1] if i > 0 else dc_low
            breakout_down = price < prev_lower and price < dc_low + 0.5 * atr
            
            if breakout_down and price_below_1d_sma:
                if vol_spike:
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
        
        # === TIME-BASED EXIT (hold at least 3 bars = 1.5 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 3:
            # Exit if trend reverses
            if position_side > 0 and price < dc_mid:
                desired_signal = 0.0
            if position_side < 0 and price > dc_mid:
                desired_signal = 0.0
        
        # === ATR TRAILING STOP ===
        if in_position and position_side > 0:
            current_stop = highest_since_entry - 2.5 * atr
            if current_stop > stop_price:
                stop_price = current_stop
                if low[i] < stop_price:
                    desired_signal = 0.0
        
        if in_position and position_side < 0:
            current_stop = lowest_since_entry + 2.5 * atr
            if current_stop < stop_price:
                stop_price = current_stop
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === CHOP EXIT FILTER ===
        if in_position and is_choppy:
            # Exit if we enter choppy regime while in position
            if bars_held >= 2:  # At least held 2 bars
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
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals