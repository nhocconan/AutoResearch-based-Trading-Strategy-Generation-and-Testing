#!/usr/bin/env python3
"""
Experiment #035: 1h Primary + 4h/1d HTF — Simplified Regime with Relaxed Entries

Hypothesis: Previous 1h strategies (#025, #030) failed with Sharpe=0.000 (0 trades).
The problem: entry conditions too strict (CRSI 15/85, multiple HTF alignment required).

Solution: LOOSEN entry thresholds significantly while keeping HTF directional bias.
1. RSI(7) with 30/70 thresholds (vs CRSI 15/85) — catches more reversals
2. CHOP > 50 = range (mean revert), CHOP < 50 = trend (follow) — simpler split
3. 4h HMA for trend direction ONLY (not strict alignment requirement)
4. Volume filter: current > 0.7x 20-bar avg (not 1.2x — too strict)
5. Session filter: 8-20 UTC only (avoid Asian session whipsaw)
6. ATR-based sizing: smaller position when vol is high (protects from 2022 crash)

Key difference from failed #025: 
- RSI 30/70 vs CRSI 15/85 (2x more signals)
- 4h HMA only vs 4h+1d+1w alignment (less restrictive)
- Volume 0.7x vs 1.2x (catches normal volume, not just spikes)
- Target: 40-70 trades/year on 1h (vs 0-10 in failed strategies)

Entry Logic:
- RANGE (CHOP>50): RSI<30 long, RSI>70 short + 4h HMA bias preference
- TREND (CHOP<50): 4h HMA slope + RSI pullback entry
- Size: 0.25 base, scale down 20% if ATR > 1.5x median
- Stop: 2.5x ATR trailing

Target: Sharpe>0.4, trades>40/symbol train, >5/symbol test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_chop_4h_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=7):
    """RSI with configurable period"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

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

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_avg[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close)) * 3600000
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # Calculate ATR median for volatility scaling
    atr_median = np.nanmedian(atr[100:]) if np.sum(~np.isnan(atr[100:])) > 0 else 1.0
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.18
    MAX_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        if not in_session:
            # Close position if outside session
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLUME FILTER (current > 0.7x average) ===
        vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 1e-10 else 0.0
        volume_ok = vol_ratio >= 0.7
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] < 50.0
        
        # === HTF TREND BIAS (4h and 1d) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 4h HMA slope (5-bar lookback)
        hma_4h_slope = 0.0
        if i >= 5 and not np.isnan(hma_4h_aligned[i-5]):
            hma_4h_slope = (hma_4h_aligned[i] - hma_4h_aligned[i-5]) / hma_4h_aligned[i-5] if abs(hma_4h_aligned[i-5]) > 1e-10 else 0.0
        
        # === VOLATILITY SCALING ===
        vol_scale = 1.0
        if atr[i] > 1.5 * atr_median:
            vol_scale = 0.8  # Reduce size in high vol
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME - Mean reversion with RSI extremes
            # Long: RSI < 30 (oversold) + prefer 4h bullish bias
            if rsi[i] < 30.0 and volume_ok:
                if hma_4h_bull or hma_1d_bull:
                    desired_signal = BASE_SIZE * vol_scale
                else:
                    desired_signal = REDUCED_SIZE * vol_scale
            
            # Short: RSI > 70 (overbought) + prefer 4h bearish bias
            elif rsi[i] > 70.0 and volume_ok:
                if hma_4h_bear or hma_1d_bear:
                    desired_signal = -BASE_SIZE * vol_scale
                else:
                    desired_signal = -REDUCED_SIZE * vol_scale
        
        elif is_trending:
            # TREND REGIME - Follow 4h trend with RSI pullback entry
            # Long: 4h bullish + RSI pullback (30-50)
            if hma_4h_bull and 30.0 <= rsi[i] <= 55.0 and volume_ok:
                desired_signal = BASE_SIZE * vol_scale
            
            # Short: 4h bearish + RSI pullback (45-70)
            elif hma_4h_bear and 45.0 <= rsi[i] <= 70.0 and volume_ok:
                desired_signal = -BASE_SIZE * vol_scale
        
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
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
        elif abs(desired_signal) >= 0.10:
            final_signal = np.sign(desired_signal) * REDUCED_SIZE
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