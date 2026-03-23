#!/usr/bin/env python3
"""
Experiment #1205: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime + Session Filter

Hypothesis: 1h strategies failed (#1195, #1198) due to either 0 trades (too strict) or too many trades (fee drag).

This version balances entry frequency with quality filters:
- 4h HMA(21) for macro trend direction (proven in best strategies)
- 1h Ehlers Fisher Transform(9) for entry timing (better reversal detection than RSI in bear markets)
- Choppiness Index(14) regime filter: >55 = range (mean revert), <45 = trend (breakout)
- Volume confirmation: volume > 0.8x 20-bar average
- Session filter: only trade 8-20 UTC (reduces overnight noise, cuts trade count)
- ATR(14) trailing stop at 2.5x

Target: 40-70 trades/year, Sharpe > 0.612
Position Size: 0.25 (lower for 1h to reduce fee impact)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_4h1d_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Catches reversals better than RSI in bear/range markets.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            normalized = ((close[i] - lowest) / price_range) - 0.5
            normalized = 0.99 * np.clip(normalized, -0.999, 0.999)
            fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            
            if i > period - 1 and not np.isnan(fisher[i-1]):
                fisher_prev[i] = fisher[i-1]
            
            fisher[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 55 = choppy/range (mean revert)
    CHOP < 45 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_session_filter(open_time):
    """
    Session filter: 8-20 UTC only (reduces overnight noise).
    Returns boolean array: True = trade allowed.
    """
    n = len(open_time)
    session = np.zeros(n, dtype=bool)
    
    for i in range(n):
        ts = pd.to_datetime(open_time[i], unit='ms')
        hour = ts.hour
        if 8 <= hour < 20:
            session[i] = True
    
    return session

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
    
    # Calculate and align 4h HMA for trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    session = calculate_session_filter(open_time)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_ma[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (4h + 1d HMA) ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        trend_1d_bull = close[i] > hma_1d_aligned[i]
        trend_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong trend = both HTF agree
        strong_bull = trend_4h_bull and trend_1d_bull
        strong_bear = trend_4h_bear and trend_1d_bear
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > 0.8 * vol_ma[i]
        
        # === SESSION FILTER ===
        in_session = session[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Bullish crossover: Fisher crosses above threshold from below
        fisher_bull_cross = (fisher_prev[i] < -1.2) and (fisher[i] >= -1.2)
        fisher_bear_cross = (fisher_prev[i] > 1.2) and (fisher[i] <= 1.2)
        
        # Extreme levels for mean reversion
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TRENDING REGIME: Fisher Crossover + HTF Trend ===
        if is_trending:
            # Long: bullish Fisher cross + strong bull trend + volume
            if fisher_bull_cross and strong_bull and vol_ok:
                desired_signal = BASE_SIZE
            # Short: bearish Fisher cross + strong bear trend + volume
            elif fisher_bear_cross and strong_bear and vol_ok:
                desired_signal = -BASE_SIZE
        
        # === CHOPPY REGIME: Mean Reversion at Fisher Extremes ===
        elif is_choppy:
            # Long: oversold Fisher + not strongly bearish
            if fisher_oversold and not strong_bear:
                desired_signal = BASE_SIZE
            # Short: overbought Fisher + not strongly bullish
            elif fisher_overbought and not strong_bull:
                desired_signal = -BASE_SIZE
        
        # === TRANSITION ZONE: Require stronger signals ===
        else:
            # Only enter on extreme Fisher + HTF alignment
            if fisher_oversold and trend_4h_bull:
                desired_signal = BASE_SIZE
            elif fisher_overbought and trend_4h_bear:
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
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals