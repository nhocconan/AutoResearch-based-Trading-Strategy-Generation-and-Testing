#!/usr/bin/env python3
"""
Experiment #010: 12h Donchian Breakout + Volume + Choppiness Regime + 1d Trend

HYPOTHESIS: Donchian breakouts mark institutional moves, but ONLY in trending
markets (CHOP < 50). Adding Choppiness Index as primary filter prevents 
whipsaws in range markets. 1d HMA adds trend direction bias. 12h timeframe
balances signal quality with reasonable trade frequency (~20-40/year).

WHY IT WORKS IN BOTH MARKETS:
- Bull: Breakout above 12h Donchian(20) + vol spike + CHOP < 50 + 1d bullish
- Bear: Breakdown below 12h Donchian(20) + vol spike + CHOP < 50 + 1d bearish
- Range (CHOP > 50): No trades — this is the key improvement over previous versions

KEY IMPROVEMENTS over failed strategies:
1. Choppiness Index filter (primary) — eliminates range market trades
2. Stricter volume threshold (1.5x) — reduces false breakouts  
3. Minimum holding period (4 bars) — prevents exit noise
4. 1d HMA trend alignment — stays with institutional flow

TIMEFRAME: 12h primary
HTF: 1d for trend bias
TARGET: 50-100 total trades over 4 years (12-25/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_chop_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = range-bound market (mean reversion)
    CHOP < 38.2 = trending market (momentum)
    """
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Sum of true range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else 0)
            tr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        hl_range = hh - ll
        
        if hl_range > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100 * (np.log10(tr_sum) / np.log10(hl_range))
    
    return chop

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
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    mid = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def calculate_rsi(close, period=14):
    """RSI"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1d HMA slope for trend direction
    hma_1d_slope_raw = calculate_hma(df_1d['close'].values, period=8)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Previous close for breakout detection
    close_prev = np.roll(close, 1)
    close_prev[0] = np.nan
    
    # Volume MA (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # 1d trend: check if price above 1d HMA and HMA is rising
    trend_bullish = np.zeros(n, dtype=bool)
    trend_bearish = np.zeros(n, dtype=bool)
    
    for i in range(21, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_slope_aligned[i]):
            price_above_hma = close[i] > hma_1d_aligned[i]
            hma_rising = hma_1d_slope_aligned[i] > hma_1d_slope_aligned[i-1] if i > 21 else False
            trend_bullish[i] = price_above_hma and hma_rising
            
            price_below_hma = close[i] < hma_1d_aligned[i]
            hma_falling = hma_1d_slope_aligned[i] < hma_1d_slope_aligned[i-1] if i > 21 else False
            trend_bearish[i] = price_below_hma and hma_falling
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_since_entry = 0
    
    # Cooldown tracking
    bars_since_exit = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if key indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Increment counters
        if bars_since_exit > 0:
            bars_since_exit += 1
        
        if in_position:
            bars_since_entry += 1
        
        # === REGIME CHECK (PRIMARY FILTER) ===
        # CHOP < 50 = trending, CHOP > 61.8 = ranging (skip)
        is_trending = chop[i] < 50.0
        is_ranging = chop[i] > 61.8
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # Stricter: 1.5x average
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # True breakout: close crosses above/below previous upper/lower band
        breakout_up = (close[i] > donch_upper[i-1]) and (close_prev[i] <= donch_upper[i-1] if i > 0 else True)
        breakout_down = (close[i] < donch_lower[i-1]) and (close_prev[i] >= donch_lower[i-1] if i > 0 else True)
        
        # === RSI VALUES ===
        rsi_val = rsi[i]
        
        # === TREND CONTEXT ===
        bull_trend = trend_bullish[i]
        bear_trend = trend_bearish[i]
        
        desired_signal = 0.0
        
        if not in_position:
            # === COOLDOWN CHECK ===
            if bars_since_exit < 5:  # Wait 5 bars after exit
                signals[i] = 0.0
                continue
            
            # === NEW LONG ENTRY ===
            # Requirements: breakout up + volume spike + trending + bullish 1d
            if breakout_up and vol_spike and is_trending and bull_trend:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Requirements: breakout down + volume spike + trending + bearish 1d
            if breakout_down and vol_spike and is_trending and bear_trend:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (3 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
            bars_since_exit = 1
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        # Minimum holding period (4 bars = 2 days)
        min_holding_passed = bars_since_entry >= 4
        
        if in_position and position_side > 0 and min_holding_passed:
            # Long exit: price breaks below lower channel OR RSI < 35 OR ranging
            if close[i] < donch_lower[i]:
                exit_triggered = True
            if rsi_val < 35:
                exit_triggered = True
            if is_ranging:
                exit_triggered = True
        
        if in_position and position_side < 0 and min_holding_passed:
            # Short exit: price breaks above upper channel OR RSI > 65 OR ranging
            if close[i] > donch_upper[i]:
                exit_triggered = True
            if rsi_val > 65:
                exit_triggered = True
            if is_ranging:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
            bars_since_exit = 1
        
        # === TAKE PROFIT (2.5R) ===
        if in_position and position_side > 0:
            profit_target = entry_price + 2.5 * entry_atr
            if high[i] >= profit_target:
                # Half position
                desired_signal = SIZE / 2
        
        if in_position and position_side < 0:
            profit_target = entry_price - 2.5 * entry_atr
            if low[i] <= profit_target:
                # Half position
                desired_signal = -SIZE / 2
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                bars_since_entry = 0
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
            else:
                # Same direction - maintain (no churn)
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                entry_bar = i
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_since_entry = 0
        
        signals[i] = desired_signal
    
    return signals