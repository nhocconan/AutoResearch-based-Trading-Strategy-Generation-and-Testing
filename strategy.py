#!/usr/bin/env python3
"""
Experiment #1378: 30m Primary + 4h/1d HTF — HTF Trend Direction + 30m Pullback Entry

Hypothesis: Lower TF (30m) strategies fail due to too many trades → fee drag.
Solution: Use 4h/1d HMA for SIGNAL DIRECTION (only trade WITH HTF trend),
30m only for ENTRY TIMING (pullback within HTF trend). This gives HTF trade
frequency (30-60/year) with 30m execution precision.

Key design:
1. 1d HMA(21) = ultra-macro bias (soft filter, only avoid counter-trend)
2. 4h HMA(21) = primary trend direction (MANDATORY alignment for entry)
3. 30m HMA(16) + RSI(7) = pullback entry timing within HTF trend
4. Volume filter = only trade on >0.8x avg volume bars
5. Session filter = only 8-20 UTC (London/NY overlap, highest liquidity)
6. ATR(14) trailing stop 2.5x = risk management
7. Position size 0.22 = conservative for 30m volatility (smaller than 4h/12h)
8. THREE entry paths per direction = ensures >=30 trades/train without over-trading

Target: 30-60 trades/year, Sharpe > 0.618, trades >= 30 train, >= 3 test
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_4h1d_pullback_session_vol_atr_v1"
timeframe = "30m"
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

def calculate_hma_slope(hma, lookback=5):
    """HMA slope - positive = uptrend, negative = downtrend"""
    n = len(hma)
    slope = np.full(n, np.nan)
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i - lookback]):
            slope[i] = (hma[i] - hma[i - lookback]) / hma[i - lookback] * 100.0
    return slope

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
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ma(volume, period=20):
    """Volume moving average for volume filter"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds, convert to seconds then to hour
    return (open_time // 1000 // 3600) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    hma_30m = calculate_hma(close, period=16)
    hma_30m_slope = calculate_hma_slope(hma_30m, lookback=3)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for pullback detection
    atr = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22
    
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
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_30m[i]) or np.isnan(hma_30m_slope[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (>0.8x average) ===
        volume_ok = volume[i] > 0.8 * vol_ma[i]
        
        # === MACRO TREND (1d HMA) - soft filter only ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) - MANDATORY alignment ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 30m PULLBACK DETECTION (entry timing) ===
        # Long: 4h bullish + 30m pulled back (RSI < 45 but > 25)
        pullback_long = rsi[i] < 45.0 and rsi[i] > 25.0
        # Short: 4h bearish + 30m rallied (RSI > 55 but < 75)
        pullback_short = rsi[i] > 55.0 and rsi[i] < 75.0
        
        # === 30m TREND CONFIRMATION ===
        trend_30m_bull = close[i] > hma_30m[i] and hma_30m_slope[i] > -0.05
        trend_30m_bear = close[i] < hma_30m[i] and hma_30m_slope[i] < 0.05
        
        # === DESIRED SIGNAL - THREE ENTRY PATHS PER DIRECTION ===
        desired_signal = 0.0
        
        # LONG ENTRY PATHS (all require 4h bullish + session + volume)
        if trend_4h_bull and in_session and volume_ok:
            # Path 1: 4h bullish + 30m pullback (RSI 25-45) + 30m turning up
            if pullback_long and hma_30m_slope[i] > -0.02:
                desired_signal = BASE_SIZE
            # Path 2: 4h bullish + 30m above HMA + RSI recovering (45-55)
            elif trend_30m_bull and 45.0 <= rsi[i] <= 55.0:
                desired_signal = BASE_SIZE * 0.7
            # Path 3: 4h bullish + macro bullish + 30m strong momentum (RSI > 55)
            elif macro_bull and rsi[i] > 55.0 and trend_30m_bull:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY PATHS (all require 4h bearish + session + volume)
        elif trend_4h_bear and in_session and volume_ok:
            # Path 1: 4h bearish + 30m rally (RSI 55-75) + 30m turning down
            if pullback_short and hma_30m_slope[i] < 0.02:
                desired_signal = -BASE_SIZE
            # Path 2: 4h bearish + 30m below HMA + RSI weakening (45-55)
            elif trend_30m_bear and 45.0 <= rsi[i] <= 55.0:
                desired_signal = -BASE_SIZE * 0.7
            # Path 3: 4h bearish + macro bearish + 30m strong momentum (RSI < 45)
            elif macro_bear and rsi[i] < 45.0 and trend_30m_bear:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
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