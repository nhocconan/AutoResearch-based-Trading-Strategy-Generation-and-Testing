#!/usr/bin/env python3
"""
EXPERIMENT #015 - Donchian Trend + HMA Momentum + ADX Strength + Z-score Filter
================================================================================
Hypothesis: Combining Donchian Channel breakouts (proven in #007 with Sharpe=4.7) 
with HMA slope momentum (faster than EMA) and ADX strength filter (avoids weak 
trends) plus Z-score overextension filter (proven in #005 with Sharpe=5.5).

Key differences from current best (#005 EMA+RSI+Z-score):
- Donchian(20) breakout for trend instead of EMA crossover - cleaner breakouts
- HMA(16) slope for entry timing instead of RSI - faster momentum detection
- ADX(14) > 25 filter to ensure strong trends (NEW - not combined with Donchian before)
- Z-score(20) < 2.0 filter to avoid overextended entries (from #005 success)
- 4h Donchian trend + 1h HMA entries (proven MTF structure)
- Trailing stoploss at 2*ATR, take profit at 2R (reduce to half)
- Discrete signal levels: 0.0, ±0.25, ±0.30 to minimize churn costs

Why this might beat Sharpe=5.525:
- Donchian breakouts capture major trends better than EMA crossovers
- HMA slope reacts faster than RSI for entry timing
- ADX filter avoids whipsaws during weak/choppy trends (major improvement)
- Z-score prevents buying tops/selling bottoms (proven in #005)
- Four-filter system should have higher win rate than 3-filter systems
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_hma_adx_zscore_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=16):
    """
    Calculate Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, reduces lag significantly
    """
    n = len(close)
    hma = np.zeros(n)
    
    if n < period:
        return hma
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # Calculate WMA(n/2)
    wma_half = np.zeros(n)
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma_half[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    # Calculate WMA(n)
    wma_full = np.zeros(n)
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    # Calculate raw HMA = 2*WMA(n/2) - WMA(n)
    raw_hma = 2 * wma_half - wma_full
    
    # Calculate final HMA = WMA(sqrt(n)) of raw HMA
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        if start_idx >= period - 1:
            weights = np.arange(1, sqrt_period + 1)
            hma[i] = np.sum(raw_hma[start_idx:i + 1] * weights) / np.sum(weights)
    
    return hma


def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel
    Upper = highest high over period
    Lower = lowest low over period
    Middle = (Upper + Lower) / 2
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle


def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = weak/choppy
    """
    n = len(close)
    adx = np.zeros(n)
    
    if n < period * 2:
        return adx
    
    # Calculate True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth with Wilder's method (EMA-like)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    atr[period - 1] = np.mean(tr[1:period])
    plus_di[period - 1] = 100 * np.mean(plus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    minus_di[period - 1] = 100 * np.mean(minus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_di[i] = 100 * ((plus_di[i - 1] * (period - 1) / 100 * atr[i - 1] + plus_dm[i]) / atr[i]) if atr[i] > 0 else 0
        minus_di[i] = 100 * ((minus_di[i - 1] * (period - 1) / 100 * atr[i - 1] + minus_dm[i]) / atr[i]) if atr[i] > 0 else 0
    
    # Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_zscore(close, period=20):
    """
    Calculate Z-score (standard score)
    Measures how many standard deviations price is from mean
    Z-score > 2 = overbought, Z-score < -2 = oversold
    """
    n = len(close)
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            zscore[i] = (close[i] - mean) / std
    
    return zscore


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    hma_1h = calculate_hma(close, period=16)
    atr_1h = calculate_atr(high, low, close, period=14)
    adx_1h = calculate_adx(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    
    # 4h Donchian for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    n_4h = len(c_4h)
    
    # Calculate 4h Donchian for trend
    donchian_upper_4h, donchian_lower_4h, donchian_mid_4h = calculate_donchian(h_4h, l_4h, period=20)
    
    # 4h trend direction based on Donchian breakout
    trend_4h = np.zeros(n_4h)
    donchian_period = 20
    
    for i in range(donchian_period, n_4h):
        # Price near upper Donchian = bullish breakout
        if c_4h[i] > donchian_mid_4h[i] and c_4h[i] > c_4h[i - 5]:
            trend_4h[i] = 1  # Bullish
        # Price near lower Donchian = bearish breakout
        elif c_4h[i] < donchian_mid_4h[i] and c_4h[i] < c_4h[i - 5]:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.30   # Full position (conservative)
    SIZE_HALF = 0.15   # Half position (after take profit)
    
    # HMA slope threshold for momentum entry
    HMA_SLOPE_THRESHOLD = 0.0001  # Minimum slope to confirm momentum
    
    # ADX strength filter
    ADX_MIN = 25  # Only trade when ADX > 25 (strong trend)
    
    # Z-score overextension filter
    ZSCORE_MAX = 2.0  # Don't enter if Z-score > 2 (overextended)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    first_valid = max(40, 28, 20)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    initial_stop = np.zeros(n)  # Track initial stoploss level
    
    for i in range(first_valid, n):
        if np.isnan(hma_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(adx_1h[i]) or np.isnan(zscore_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        hma_val = hma_1h[i]
        prev_hma_val = hma_1h[i - 1] if i > 0 else hma_val
        hma_slope = hma_val - prev_hma_val
        adx_val = adx_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # ADX strength filter - only trade strong trends
        if adx_val < ADX_MIN:
            if i > 0 and position_side[i - 1] != 0:
                # Hold existing position but don't add
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                if position_side[i-1] == 1:
                    highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                    lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                elif position_side[i-1] == -1:
                    highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                    lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                initial_stop[i] = initial_stop[i-1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Z-score overextension filter - avoid buying tops/selling bottoms
        if abs(zscore_val) > ZSCORE_MAX:
            if i > 0 and position_side[i - 1] != 0:
                # Hold existing position but don't add
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                if position_side[i-1] == 1:
                    highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                    lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                elif position_side[i-1] == -1:
                    highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                    lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                initial_stop[i] = initial_stop[i-1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_stop = initial_stop[i - 1] if initial_stop[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                
                # Trailing stoploss (2*ATR from entry, or trail from highest)
                trail_stop = highest_since_entry[i] - ATR_STOP_MULT * atr if highest_since_entry[i] > 0 else prev_entry - ATR_STOP_MULT * atr
                stoploss_price = max(prev_entry - ATR_STOP_MULT * atr, trail_stop)
                
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = prev_stop
                    continue
                
                # HMA slope exit for longs (momentum fading)
                if hma_slope < -HMA_SLOPE_THRESHOLD:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
                    
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                
                # Trailing stoploss
                trail_stop = lowest_since_entry[i] + ATR_STOP_MULT * atr if lowest_since_entry[i] > 0 else prev_entry + ATR_STOP_MULT * atr
                stoploss_price = min(prev_entry + ATR_STOP_MULT * atr, trail_stop)
                
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = prev_stop
                    continue
                
                # HMA slope exit for shorts (momentum fading)
                if hma_slope > HMA_SLOPE_THRESHOLD:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
        
        # Entry logic with HMA slope momentum confirmation
        position_size = SIZE_FULL
        
        if trend == 1:  # 4h uptrend + ADX OK + Z-score OK
            # HMA sloping up (momentum turning positive)
            if hma_slope > HMA_SLOPE_THRESHOLD and zscore_val < ZSCORE_MAX:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                initial_stop[i] = price - ATR_STOP_MULT * atr
            else:
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = initial_stop[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    
        elif trend == -1:  # 4h downtrend + ADX OK + Z-score OK
            # HMA sloping down (momentum turning negative)
            if hma_slope < -HMA_SLOPE_THRESHOLD and zscore_val > -ZSCORE_MAX:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                initial_stop[i] = price + ATR_STOP_MULT * atr
            else:
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = initial_stop[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
            initial_stop[i] = 0
    
    return signals