#!/usr/bin/env python3
"""
Experiment #1390: 1h Primary + 4h/12h HTF — Pullback Within Trend with Session/Volume Filter

Hypothesis: Previous 1h/30m failures (#1380, #1385, #1388) had Sharpe<0 or 0 trades due to
over-filtering with conflicting regime switches. The key insight from successful higher-TF
strategies: use HTF for TREND DIRECTION, primary TF only for ENTRY TIMING within that trend.

Design:
1. 12h HMA(21) = macro trend bias (long only if price>12h HMA, short if price<12h HMA)
2. 4h HMA(21) + slope = primary trend confirmation (must align with 12h)
3. 1h RSI(14) pullback = entry trigger (RSI 35-45 for long, 55-65 for short within trend)
4. Session filter = only 8-20 UTC (high liquidity hours, avoid Asian chop)
5. Volume filter = current volume > 0.8x 20-bar avg (confirm participation)
6. ATR(14) trailing stop 2.5x = risk management
7. FOUR entry paths per direction = ensures >=30 trades/train, >=5 test
8. Position size 0.25 = conservative for 1h volatility (fee drag control)

Why this differs from failed #1380/#1385:
- No regime switching (Choppiness/CRSI caused 0 trades)
- Pullback entries (not breakout) = higher win rate in trending markets
- Session filter reduces noise without eliminating all signals
- Multiple entry paths ensure trade frequency while maintaining quality

Target: 40-80 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_4h12h_pullback_session_vol_atr_v3"
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

def calculate_hma_slope(hma, lookback=5):
    """HMA slope - positive = uptrend, negative = downtrend"""
    n = len(hma)
    slope = np.full(n, np.nan)
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i - lookback]):
            slope[i] = (hma[i] - hma[i - lookback]) / hma[i - lookback] * 100.0
    return slope

def calculate_rsi(close, period=14):
    """Relative Strength Index - for pullback entry timing"""
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

def calculate_volume_avg(volume, period=20):
    """Rolling average volume for volume confirmation filter"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time_array):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array / 1000 / 3600) % 24).astype(int)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMAs for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 4h HMA slope for trend strength
    hma_4h_slope_raw = calculate_hma_slope(hma_4h_raw, lookback=3)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
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
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_slope_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (12h HMA) - direction bias ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === PRIMARY TREND (4h HMA + slope) - must align with macro ===
        trend_4h_bull = close[i] > hma_4h_aligned[i] and hma_4h_slope_aligned[i] > 0.0
        trend_4h_bear = close[i] < hma_4h_aligned[i] and hma_4h_slope_aligned[i] < 0.0
        
        # === TREND CONFLUENCE (both 12h and 4h agree) ===
        strong_bull = macro_bull and trend_4h_bull
        strong_bear = macro_bear and trend_4h_bear
        
        # === RSI PULLBACK (entry timing within trend) ===
        # Long: RSI pulled back to 35-50 in uptrend
        # Short: RSI rallied to 50-65 in downtrend
        rsi_pullback_long = 35.0 <= rsi[i] <= 50.0
        rsi_pullback_short = 50.0 <= rsi[i] <= 65.0
        rsi_momentum_long = rsi[i] > 45.0 and rsi[i] < 60.0
        rsi_momentum_short = rsi[i] > 40.0 and rsi[i] < 55.0
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === SESSION FILTER (8-20 UTC = high liquidity) ===
        session_ok = 8 <= hours[i] <= 20
        
        # === DESIRED SIGNAL - FOUR ENTRY PATHS PER DIRECTION ===
        desired_signal = 0.0
        
        # LONG ENTRY PATHS
        # Path 1: Strong bull trend + RSI pullback + volume (primary entry)
        if strong_bull and rsi_pullback_long and volume_ok:
            desired_signal = BASE_SIZE
        # Path 2: Strong bull trend + RSI pullback + session (time-based entry)
        elif strong_bull and rsi_pullback_long and session_ok:
            desired_signal = BASE_SIZE
        # Path 3: Macro bull + 4h above HMA + RSI momentum (continuation)
        elif macro_bull and close[i] > hma_4h_aligned[i] and rsi_momentum_long and volume_ok:
            desired_signal = BASE_SIZE * 0.5
        # Path 4: Both HMAs aligned bull + RSI > 45 (trend follow)
        elif macro_bull and trend_4h_bull and rsi[i] > 45.0 and rsi[i] < 65.0:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY PATHS
        # Path 1: Strong bear trend + RSI pullback + volume (primary entry)
        elif strong_bear and rsi_pullback_short and volume_ok:
            desired_signal = -BASE_SIZE
        # Path 2: Strong bear trend + RSI pullback + session (time-based entry)
        elif strong_bear and rsi_pullback_short and session_ok:
            desired_signal = -BASE_SIZE
        # Path 3: Macro bear + 4h below HMA + RSI momentum (continuation)
        elif macro_bear and close[i] < hma_4h_aligned[i] and rsi_momentum_short and volume_ok:
            desired_signal = -BASE_SIZE * 0.5
        # Path 4: Both HMAs aligned bear + RSI < 55 (trend follow)
        elif macro_bear and trend_4h_bear and rsi[i] > 35.0 and rsi[i] < 55.0:
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