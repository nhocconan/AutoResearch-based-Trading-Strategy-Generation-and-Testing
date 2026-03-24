#!/usr/bin/env python3
"""
Experiment #1430: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume Session

Hypothesis: 1h timeframe with 4h/12h HTF trend filter will generate 40-70 trades/year
with positive Sharpe by combining proven patterns from best strategies:
1. 4h HMA(21) = primary trend direction (faster than 1d, slower than 1h noise)
2. 12h HMA(50) = macro bias confirmation (avoids counter-trend traps)
3. 1h RSI(14) pullback = entry timing within HTF trend (proven in best strategy)
4. Choppiness Index = adjust thresholds, NOT block entries (learned from failures)
5. Volume > 0.7x avg + Session 8-20 UTC = filter low-liquidity traps
6. ATR(14) 2.5x trailing stop = risk management

Why this should work (learning from #1418-#1429 failures):
- Recent 0-trade failures had TOO STRICT conditions (all filters must align perfectly)
- Solution: Use HTF trend as PRIMARY filter, 1h indicators as ENTRY TIMING only
- Relaxed RSI thresholds (35/65 not 30/70) to ensure trades trigger
- Choppiness adjusts size, not entry permission (range=0.20, trend=0.30)
- Volume filter relaxed to 0.7x (not 1.0x) to allow more trades

Position sizing: 0.25 base, 0.30 in strong trend, 0.20 in choppy
Target: 40-70 trades/year, Sharpe > 0.618 (beat current best), DD < -30%
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_trend_vol_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                        abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
                tr_sum += tr
            
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sma[i] = np.nanmean(volume[i-period+1:i+1])
    
    return vol_sma

def get_hour_from_open_time(open_time_arr):
    """Extract hour from open_time (milliseconds timestamp)"""
    hours = np.zeros(len(open_time_arr), dtype=np.int32)
    for i in range(len(open_time_arr)):
        # Convert ms to seconds, then to datetime
        ts_sec = open_time_arr[i] / 1000.0
        hours[i] = int((ts_sec % 86400) / 3600)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Extract hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA(21) for primary trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA(50) for macro bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    rsi_14 = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    TREND_SIZE = 0.30
    CHOP_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER (relaxed: > 0.7x avg) ===
        vol_ok = volume[i] > 0.7 * vol_sma_20[i]
        
        # === MACRO TREND (12h HMA) - bias filter ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === PRIMARY TREND (4h HMA) - entry direction ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === TREND CONFIRMATION (both HTF agree) ===
        strong_bull = macro_bull and trend_bull
        strong_bear = macro_bear and trend_bear
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] < 45.0
        
        # === RSI PULLBACK (relaxed thresholds for more trades) ===
        rsi_pullback_long = rsi_14[i] < 45.0 and rsi_14[i] > 30.0
        rsi_pullback_short = rsi_14[i] > 55.0 and rsi_14[i] < 70.0
        rsi_extreme_long = rsi_14[i] < 35.0
        rsi_extreme_short = rsi_14[i] > 65.0
        
        # === DESIRED SIGNAL - SIMPLIFIED LOGIC ===
        desired_signal = 0.0
        use_size = BASE_SIZE
        
        # Determine size based on regime
        if is_trending and strong_bull:
            use_size = TREND_SIZE
        elif is_trending and strong_bear:
            use_size = TREND_SIZE
        elif is_choppy:
            use_size = CHOP_SIZE
        
        # LONG ENTRIES (must have session + volume)
        if in_session and vol_ok:
            # Path 1: Strong bull trend + RSI pullback
            if strong_bull and rsi_pullback_long:
                desired_signal = use_size
            # Path 2: Strong bull trend + RSI extreme (oversold)
            elif strong_bull and rsi_extreme_long:
                desired_signal = use_size
            # Path 3: Trending regime + any bull signal + RSI < 50
            elif is_trending and trend_bull and rsi_14[i] < 50.0:
                desired_signal = BASE_SIZE
        
        # SHORT ENTRIES (must have session + volume)
        if in_session and vol_ok:
            # Path 1: Strong bear trend + RSI pullback
            if strong_bear and rsi_pullback_short:
                desired_signal = -use_size
            # Path 2: Strong bear trend + RSI extreme (overbought)
            elif strong_bear and rsi_extreme_short:
                desired_signal = -use_size
            # Path 3: Trending regime + any bear signal + RSI > 50
            elif is_trending and trend_bear and rsi_14[i] > 50.0:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.5:
            final_signal = np.sign(desired_signal) * use_size
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals