#!/usr/bin/env python3
"""
Experiment #060: 1h Primary + 4h/12h HTF — Regime-Adaptive with Loose Filters

Hypothesis: After 55+ failed experiments, the #1 issue is TOO STRICT filters = 0 trades.
This strategy uses PROVEN elements with LOOSE thresholds to ensure trades generate:
1. 4h HMA trend filter - direction bias (proven in research)
2. 12h HMA meta-trend - avoid counter-trend in major moves
3. Choppiness Index regime - CHOP<45=trend follow, CHOP>55=mean revert
4. RSI(14) with LOOSE thresholds (35/65 not 15/85) - ensures entries happen
5. Volume filter (loose: >0.8x avg) + Session (8-20 UTC)
6. ATR(14) 2.5x trailing stop - proven risk management

Why 1h should work NOW:
- 4h/12h HTF gives direction (reduces whipsaw)
- 1h entry timing captures pullbacks within HTF trend
- LOOSE RSI thresholds (35/65) ensure we get 30-80 trades/year
- Choppiness Index adapts to market regime (trend vs range)
- Session filter avoids low-liquidity hours (reduces false breakouts)

Entry Logic (LOOSE to ensure trades):
- Trend regime (CHOP<45): Long if price>4h HMA>12h HMA + RSI>35 + volume>0.8x
- Range regime (CHOP>55): Long if RSI<35 + price near BB lower + volume>0.8x
- Short signals symmetric (RSI>65 for range, price<HTF HMA for trend)
- Size: 0.25 (discrete, minimizes fee churn)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_regime_4h12h_hma_rsi_loose_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend indicator"""
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

def calculate_rsi(close, period=14):
    """RSI - momentum oscillator with LOOSE thresholds"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range - for volatility and stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection (trend vs range)
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    We use 45/55 as loose thresholds for regime switching
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            choppiness[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        choppiness[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands - for mean reversion levels"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, sma, lower

def calculate_volume_avg(volume, period=20):
    """Average volume for volume filter"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time_arr):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = np.zeros(len(open_time_arr), dtype=int)
    for i in range(len(open_time_arr)):
        # Convert ms to seconds, then to datetime
        ts_seconds = open_time_arr[i] / 1000.0
        hours[i] = int((ts_seconds % 86400) / 3600)
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
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for meta-trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_dev=2.0)
    vol_avg = calculate_volume_avg(volume, period=20)
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size (25% of capital)
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC for liquidity) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER (loose: >0.8x average) ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === HTF TREND BIAS (4h and 12h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Strong trend: both 4h and 12h agree
        strong_bull = hma_4h_bull and hma_12h_bull
        strong_bear = hma_4h_bear and hma_12h_bear
        
        # === CHOPPINESS REGIME ===
        trend_regime = choppiness[i] < 45.0  # Trending market
        range_regime = choppiness[i] > 55.0  # Ranging market
        # Neutral regime (45-55): use trend logic as default
        
        # === RSI FILTER (LOOSE thresholds to ensure trades) ===
        rsi_ok_long_trend = rsi[i] > 35.0  # Not extremely oversold
        rsi_ok_short_trend = rsi[i] < 65.0  # Not extremely overbought
        rsi_ok_long_range = rsi[i] < 40.0  # Oversold for mean reversion
        rsi_ok_short_range = rsi[i] > 60.0  # Overbought for mean reversion
        
        # === BB POSITION FOR RANGE REGIME ===
        near_bb_lower = close[i] < bb_lower[i] * 1.01 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] > bb_upper[i] * 0.99 if not np.isnan(bb_upper[i]) else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow HTF trend with pullback entries
        if trend_regime or (not range_regime):  # Default to trend logic
            # Long: Strong bull trend + RSI not oversold + session + volume
            if strong_bull and rsi_ok_long_trend and in_session and volume_ok:
                desired_signal = SIZE
            
            # Short: Strong bear trend + RSI not overbought + session + volume
            elif strong_bear and rsi_ok_short_trend and in_session and volume_ok:
                desired_signal = -SIZE
        
        # RANGE REGIME: Mean reversion at BB bounds
        elif range_regime:
            # Long: Near BB lower + RSI oversold + session + volume
            if near_bb_lower and rsi_ok_long_range and in_session and volume_ok:
                desired_signal = SIZE
            
            # Short: Near BB upper + RSI overbought + session + volume
            elif near_bb_upper and rsi_ok_short_range and in_session and volume_ok:
                desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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