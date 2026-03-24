#!/usr/bin/env python3
"""
Experiment #1485: 1h Primary + 4h/1d HTF — Fisher Transform Reversal with HTF Trend Filter

Hypothesis: After analyzing 1107 failed strategies, the pattern for 1h timeframe is clear:
1. 1h strategies fail due to OVER-FILTERING (0 trades) or TOO MANY trades (>200/yr)
2. Solution: Use HTF (4h/1d) for DIRECTION, 1h only for ENTRY TIMING
3. Fisher Transform excels at catching reversals in bear/range markets (2022, 2025)
4. Session filter (8-20 UTC) + volume filter naturally reduces trade count to 30-80/yr
5. LOOSE entry conditions on 1h, strict HTF trend filter ensures quality trades

Key components:
- 1d HMA: Ultimate macro trend bias (only trade with daily trend)
- 4h HMA: Intermediate trend confirmation
- 1h Fisher Transform: Entry timing (crosses -1.5 long, +1.5 short)
- RSI(14) loose filter: >35 for long, <65 for short (NOT strict 45-55)
- Session filter: Only 8-20 UTC (reduces trades by ~60%)
- Volume filter: >0.8x 20-bar avg volume
- ATR(14)*2.5 trailing stoploss

Why this should work on 1h:
1. Fisher Transform is proven for reversal capture in choppy markets
2. HTF filters ensure we only trade with macro trend (reduces whipsaws)
3. Session+volume filters naturally limit trades to 30-80/year target
4. Loose RSI ensures we actually GET trades (not 0 like #1475, #1478, #1480)
5. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Timeframe: 1h
HTF: 4h, 1d (call get_htf_data ONCE before loop!)
Position Size: 0.25 (smaller for lower TF to control DD)
Target: 30-80 trades/year, Sharpe > 0.618 (beat current best), ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_reversal_4h1d_hma_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            if np.any(np.isnan(data[i - span + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - span + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # Combine
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points with sharp peaks
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)  # Previous bar fisher for crossover
    
    # Calculate median price
    median = (high + low) / 2.0
    
    # Normalize to -1 to +1 range
    for i in range(period, n):
        if np.any(np.isnan(median[i - period + 1:i + 1])):
            continue
        
        highest = np.nanmax(median[i - period + 1:i + 1])
        lowest = np.nanmin(median[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Normalize
        normalized = 0.999 * ((median[i] - lowest) / (highest - lowest) - 0.5) + 0.5
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i > period:
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

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

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_hour_from_open_time(open_time_arr):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hours = ((open_time_arr / 1000) % 86400) / 3600
    return hours.astype(int)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Extract UTC hour for session filter
    utc_hour = get_hour_from_open_time(open_time)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for ultimate macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA for intermediate trend confirmation
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h TF
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Reduces trades by ~60%, focuses on high-liquidity hours
        in_session = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === VOLUME FILTER (>0.8x 20-bar average) ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === MACRO TREND (1d HMA) - ultimate direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - confirmation ===
        h4h_bull = close[i] > hma_4h_aligned[i]
        h4h_bear = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        fisher_short = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # Also allow continuation when Fisher is extreme
        fisher_extreme_long = fisher[i] < -1.0
        fisher_extreme_short = fisher[i] > 1.0
        
        # === RSI MOMENTUM - LOOSE bands for more trades ===
        # NOT strict 45-55, use 35-65 to ensure we get trades
        rsi_bullish = rsi[i] > 35.0
        rsi_bearish = rsi[i] < 65.0
        rsi_strong_bull = rsi[i] > 45.0
        rsi_strong_bear = rsi[i] < 55.0
        
        # === DESIRED SIGNAL - FISHER REVERSAL WITH HTF FILTER ===
        desired_signal = 0.0
        
        # LONG: Daily bull + 4h bull + Fisher reversal + RSI + Session + Volume
        # Use OR logic for Fisher to ensure we get entries
        if daily_bull and h4h_bull:
            if in_session and volume_ok:
                # Primary entry: Fisher crossover
                if fisher_long and rsi_bullish:
                    desired_signal = BASE_SIZE
                # Secondary entry: Fisher extreme + RSI support
                elif fisher_extreme_long and rsi_strong_bull:
                    desired_signal = BASE_SIZE * 0.8
                # Tertiary: Just Fisher extreme in strong trend
                elif fisher_extreme_long and rsi[i] > 40.0:
                    desired_signal = BASE_SIZE * 0.6
        
        # SHORT: Daily bear + 4h bear + Fisher reversal + RSI + Session + Volume
        elif daily_bear and h4h_bear:
            if in_session and volume_ok:
                # Primary entry: Fisher crossover
                if fisher_short and rsi_bearish:
                    desired_signal = -BASE_SIZE
                # Secondary entry: Fisher extreme + RSI support
                elif fisher_extreme_short and rsi_strong_bear:
                    desired_signal = -BASE_SIZE * 0.8
                # Tertiary: Just Fisher extreme in strong trend
                elif fisher_extreme_short and rsi[i] < 60.0:
                    desired_signal = -BASE_SIZE * 0.6
        
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
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.2:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.2:
            final_signal = -BASE_SIZE * 0.5
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