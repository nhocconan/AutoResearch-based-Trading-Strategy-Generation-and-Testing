#!/usr/bin/env python3
"""
Experiment #738: 30m Primary + 4h/1d HTF — RSI Pullback within HTF Trend

Hypothesis: After 494 failed strategies, the pattern is clear — complex regime detection
(Choppiness + CRSI) causes 0 trades. Breakout strategies on 30m generate TOO MANY trades.

NEW APPROACH for 30m:
1. 1d HMA(21) for regime bias (bull/bear)
2. 4h HMA(21) for trend direction (proven in best strategies)
3. 30m RSI(14) pullback entries WITHIN HTF trend (not breakout = fewer trades)
4. Volume filter: only trade when volume > 1.2x 20-bar avg (filters false signals)
5. Session filter: only 8-20 UTC (high liquidity hours)
6. ATR(14) trailing stop 2.5x for risk management
7. Discrete signal sizes: 0.0, ±0.25 (smaller for lower TF)

Key insight: 30m breakout = too many trades. 30m pullback within 4h trend = fewer, higher quality trades.
Target: 30-80 trades/year, Sharpe > 0.612, ALL symbols positive
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_hma_4h1d_vol_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(prices):
    """Extract hour from open_time for session filtering."""
    # open_time is in milliseconds since epoch
    # Convert to datetime and extract hour (UTC)
    try:
        hours = (prices['open_time'].values // 3600000) % 24
        return hours
    except:
        # Fallback if open_time not available
        return np.full(len(prices), 12)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    sma_50_30m = calculate_sma(close, period=50)
    sma_200_30m = calculate_sma(close, period=200)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    hours = get_hour_from_open_time(prices)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for lower TF
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50_30m[i]) or np.isnan(sma_200_30m[i]) or np.isnan(vol_sma_30m[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === VOLUME FILTER (must be > 1.2x average) ===
        volume_ok = volume[i] > 1.2 * vol_sma_30m[i]
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === TREND DIRECTION (4h HTF HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50_30m[i]
        above_sma200 = close[i] > sma_200_30m[i]
        below_sma50 = close[i] < sma_50_30m[i]
        below_sma200 = close[i] < sma_200_30m[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (RSI pullback within uptrend) ===
        long_signal = False
        
        # Confluence 1: 1d bullish + 4h bullish + RSI pullback (35-50) + volume + session
        if (trend_1d_bullish and trend_4h_bullish and 
            rsi_30m[i] >= 35 and rsi_30m[i] <= 50 and
            in_session and volume_ok):
            long_signal = True
        
        # Confluence 2: 1d bullish + 4h bullish + price > SMA50 + RSI not overbought
        if (trend_1d_bullish and trend_4h_bullish and above_sma50 and
            rsi_30m[i] >= 40 and rsi_30m[i] <= 55 and
            in_session and volume_ok):
            long_signal = True
        
        # Confluence 3: Strong trend (above both SMA) + 4h bullish + RSI pullback
        if (above_sma50 and above_sma200 and trend_4h_bullish and
            rsi_30m[i] >= 35 and rsi_30m[i] <= 50 and
            in_session and volume_ok):
            long_signal = True
        
        if long_signal:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS (RSI rally within downtrend) ===
        short_signal = False
        
        # Confluence 1: 1d bearish + 4h bearish + RSI rally (50-65) + volume + session
        if (trend_1d_bearish and trend_4h_bearish and 
            rsi_30m[i] >= 50 and rsi_30m[i] <= 65 and
            in_session and volume_ok):
            short_signal = True
        
        # Confluence 2: 1d bearish + 4h bearish + price < SMA50 + RSI not oversold
        if (trend_1d_bearish and trend_4h_bearish and below_sma50 and
            rsi_30m[i] >= 45 and rsi_30m[i] <= 60 and
            in_session and volume_ok):
            short_signal = True
        
        # Confluence 3: Strong downtrend (below both SMA) + 4h bearish + RSI rally
        if (below_sma50 and below_sma200 and trend_4h_bearish and
            rsi_30m[i] >= 50 and rsi_30m[i] <= 65 and
            in_session and volume_ok):
            short_signal = True
        
        if short_signal:
            desired_signal = -BASE_SIZE
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with 1d HMA trend
        if long_signal and short_signal:
            if trend_1d_bullish:
                desired_signal = BASE_SIZE
            elif trend_1d_bearish:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h HMA still bullish and RSI not extremely overbought
                if trend_4h_bullish and rsi_30m[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h HMA still bearish and RSI not extremely oversold
                if trend_4h_bearish and rsi_30m[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses or RSI extremely overbought
            if trend_4h_bearish or rsi_30m[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses or RSI extremely oversold
            if trend_4h_bullish or rsi_30m[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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
        
        signals[i] = desired_signal
    
    return signals