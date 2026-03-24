#!/usr/bin/env python3
"""
Experiment #1486: 12h Primary + 1d HTF — HMA Trend + Donchian Breakout + RSI Momentum

Hypothesis: Based on #1482 (12h HMA+Donchian+RSI Sharpe=0.237) and #1477 (1d simple trend Sharpe=0.150),
higher timeframes with simple trend-following logic work best. This strategy combines:
- 1d HMA for macro trend bias (only trade in direction of daily trend)
- 12h HMA(21) for primary trend direction
- Donchian(20) breakout for entry timing
- RSI(14) momentum filter (loose bands 45-55 for sufficient trades)
- ATR(14)*2.5 trailing stoploss

Why 12h + 1d should beat current best (Sharpe=0.618):
1. 12h = target 20-50 trades/year (minimal fee drag ~1-2.5%)
2. 1d HMA filter prevents trading against macro trend (critical for 2022 crash)
3. HMA(21) smoother than EMA, less whipsaw in chop
4. Donchian breakout catches momentum moves without lag
5. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
6. ATR trailing stop protects from reversals

Timeframe: 12h
HTF: 1d (call get_htf_data ONCE before loop!)
Position Size: 0.30 (discrete levels: 0.0, ±0.25, ±0.30)
Target: 20-50 trades/year, Sharpe > 0.618, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_donchian_rsi_1d_atr_v2"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            if np.all(~np.isnan(series[i - span + 1:i + 1])):
                result[i] = np.sum(series[i - span + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    half_period = period // 2
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # HMA calculation
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            # Apply WMA to the difference with sqrt(period)
            sqrt_period = int(np.sqrt(period))
            if i >= sqrt_period - 1:
                diff_series = np.array([2.0 * wma_half[j] - wma_full[j] if not np.isnan(wma_half[j]) and not np.isnan(wma_full[j]) else np.nan for j in range(i - sqrt_period + 1, i + 1)])
                if np.all(~np.isnan(diff_series)):
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(diff_series * weights) / np.sum(weights)
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_sma(close, period=50):
    """Simple Moving Average for additional trend filter"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - direction bias ===
        # Only trade in direction of daily trend
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA) ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === HMA SLOPE (trend strength) ===
        hma_slope_bull = hma_12h[i] > hma_12h[i-5] if not np.isnan(hma_12h[i-5]) else False
        hma_slope_bear = hma_12h[i] < hma_12h[i-5] if not np.isnan(hma_12h[i-5]) else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI MOMENTUM - LOOSE bands for more trades (45-55) ===
        rsi_bullish = rsi[i] > 45.0
        rsi_bearish = rsi[i] < 55.0
        rsi_strong_bull = rsi[i] > 50.0
        rsi_strong_bear = rsi[i] < 50.0
        
        # === SMA 50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === DESIRED SIGNAL - TREND FOLLOWING WITH BREAKOUT ===
        desired_signal = 0.0
        
        # LONG: Daily bull + 12h bull + Breakout or HMA slope + RSI support
        if daily_bull and hma_bull:
            if breakout_high and rsi_bullish:
                desired_signal = BASE_SIZE  # Strong breakout entry
            elif hma_slope_bull and above_sma50 and rsi_strong_bull:
                desired_signal = BASE_SIZE * 0.85  # Trend continuation
            elif hma_bull and rsi[i] > 48.0 and close[i] > hma_1d_aligned[i]:
                desired_signal = BASE_SIZE * 0.6  # Weaker trend follow
        
        # SHORT: Daily bear + 12h bear + Breakout or HMA slope + RSI support
        elif daily_bear and hma_bear:
            if breakout_low and rsi_bearish:
                desired_signal = -BASE_SIZE  # Strong breakdown entry
            elif hma_slope_bear and below_sma50 and rsi_strong_bear:
                desired_signal = -BASE_SIZE * 0.85  # Trend continuation
            elif hma_bear and rsi[i] < 52.0 and close[i] < hma_1d_aligned[i]:
                desired_signal = -BASE_SIZE * 0.6  # Weaker trend follow
        
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
            final_signal = BASE_SIZE * 0.85
        elif desired_signal >= BASE_SIZE * 0.2:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE * 0.85
        elif desired_signal <= -BASE_SIZE * 0.2:
            final_signal = -BASE_SIZE * 0.6
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