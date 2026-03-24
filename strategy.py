#!/usr/bin/env python3
"""
Experiment #513: 5m Primary + 15m/4h HTF — Session-Filtered Pullback Strategy

Hypothesis: 5m timeframe has ZERO prior experiments. Key insight: 5m works ONLY when
HTF provides trend direction and 5m provides entry timing. This avoids counter-trend
trades that get chopped on lower timeframes.

Strategy logic:
1. 4h HMA(21) = major trend bias (only trade in this direction)
2. 15m HMA(21) = intermediate trend confirmation
3. 15m Choppiness(14) = regime filter (avoid mean-reversion in strong trends)
4. 5m RSI(7) = pullback entry timing (RSI<35 long in uptrend, RSI>65 short in downtrend)
5. 5m Session filter = only trade 08:00-20:00 UTC (London+NY overlap, high volume)
6. 5m ATR(14)*2.0 = stoploss on all positions
7. 5m Volume filter = only trade when volume > 1.5x 20-bar average

Key design choices for 5m:
- Small position size (0.15-0.20) due to higher trade frequency
- Session filter MANDATORY (avoids low-volume Asian session whipsaws)
- HTF trend alignment REQUIRED (no counter-trend trades on 5m)
- RSI thresholds loosened (25-75 instead of 20-80) to generate enough trades
- Volume confirmation to avoid fake breakouts

Target: Sharpe>0.40, trades>=200 train (50/year), trades>=30 test
Timeframe: 5m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_rsi_pullback_15m4h_v1"
timeframe = "5m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def is_session_active(open_time, start_hour=8, end_hour=20):
    """
    Check if timestamp is within trading session (UTC)
    Default: 08:00-20:00 UTC (London + NY overlap)
    open_time is in milliseconds since epoch
    """
    # Convert milliseconds to datetime
    ts = pd.to_datetime(open_time, unit='ms', utc=True)
    hour = ts.hour
    
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for major trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 15m HMA for intermediate trend
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    # Calculate and align 15m Choppiness for regime detection
    chop_15m_raw = calculate_choppiness(
        df_15m['high'].values,
        df_15m['low'].values,
        df_15m['close'].values,
        period=14
    )
    chop_15m_aligned = align_htf_to_ltf(prices, df_15m, chop_15m_raw)
    
    # Calculate 5m indicators
    hma_5m = calculate_hma(close, period=21)
    rsi_5m = calculate_rsi(close, period=7)  # Faster RSI for 5m entries
    atr_5m = calculate_atr(high, low, close, period=14)
    vol_sma_5m = calculate_volume_sma(volume, period=20)
    sma_50_5m = calculate_sma(close, 50)
    sma_200_5m = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_5m[i]) or atr_5m[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_5m[i]) or np.isnan(rsi_5m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_15m_aligned[i]) or np.isnan(vol_sma_5m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (MANDATORY for 5m) ===
        in_session = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === 4h HTF MAJOR TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m HTF INTERMEDIATE TREND ===
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        
        # === 15m CHOPPINESS REGIME ===
        chop_trend = chop_15m_aligned[i] < 45.0  # Trending market
        chop_range = chop_15m_aligned[i] > 55.0  # Range-bound market
        # chop between 45-55 = neutral
        
        # === 5m RSI PULLBACK ===
        rsi_oversold = rsi_5m[i] < 35.0  # Loosened for more trades
        rsi_overbought = rsi_5m[i] > 65.0
        rsi_extreme_oversold = rsi_5m[i] < 25.0
        rsi_extreme_overbought = rsi_5m[i] > 75.0
        
        # RSI recovery/decline
        rsi_rising = rsi_5m[i] > rsi_5m[i-1] if i > 0 else False
        rsi_falling = rsi_5m[i] < rsi_5m[i-1] if i > 0 else False
        
        # === 5m VOLUME FILTER ===
        vol_ratio = volume[i] / vol_sma_5m[i] if vol_sma_5m[i] > 1e-10 else 1.0
        vol_confirmed = vol_ratio > 1.2  # Volume 20% above average
        
        # === 5m HMA TREND ===
        hma_5m_bull = close[i] > hma_5m[i]
        hma_5m_bear = close[i] < hma_5m[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50_5m[i] if not np.isnan(sma_50_5m[i]) else False
        below_sma50 = close[i] < sma_50_5m[i] if not np.isnan(sma_50_5m[i]) else False
        above_sma200 = close[i] > sma_200_5m[i] if not np.isnan(sma_200_5m[i]) else False
        below_sma200 = close[i] < sma_200_5m[i] if not np.isnan(sma_200_5m[i]) else False
        
        # === TREND ALIGNMENT SCORE ===
        # Count how many HTF agree on direction
        bull_alignment = sum([htf_4h_bull, htf_15m_bull, hma_5m_bull])
        bear_alignment = sum([htf_4h_bear, htf_15m_bear, hma_5m_bear])
        
        strong_bull = bull_alignment >= 2
        strong_bear = bear_alignment >= 2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # TREND REGIME: Pullback entries in HTF trend direction
        if chop_trend or (not chop_range and not chop_trend):
            # LONG: HTF bullish + RSI pullback + volume confirmation
            if strong_bull and rsi_oversold and rsi_rising and vol_confirmed and above_sma50:
                desired_signal = SIZE_STRONG
            elif strong_bull and rsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE
            # SHORT: HTF bearish + RSI pullback + volume confirmation
            elif strong_bear and rsi_overbought and rsi_falling and vol_confirmed and below_sma50:
                desired_signal = -SIZE_STRONG
            elif strong_bear and rsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at extremes (only with HTF agreement)
        if chop_range:
            # LONG: HTF not strongly bearish + extreme oversold
            if not strong_bear and rsi_extreme_oversold and vol_confirmed:
                desired_signal = SIZE_BASE * 0.8
            # SHORT: HTF not strongly bullish + extreme overbought
            elif not strong_bull and rsi_extreme_overbought and vol_confirmed:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_5m[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals