#!/usr/bin/env python3
"""
EXPERIMENT #018 - KAMA Adaptive Trend + MACD Momentum + Bollinger Regime Filter
===============================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA,
reducing whipsaw in choppy markets. MACD histogram crosses catch momentum shifts earlier
than RSI levels. Bollinger Band Width detects squeeze/expansion regimes to avoid low-vol
traps. 4H EMA(21/55) crossover provides robust trend filter across all assets.

Key innovations vs mtf_donchian_hma_rsi_zscore_v1:
- KAMA(21) instead of HMA - adapts to market efficiency ratio
- MACD histogram cross for entry timing instead of RSI pullback
- Bollinger Band Width percentile for regime filter instead of Z-score
- 4H EMA(21/55) crossover trend filter (more robust than Donchian)
- Volume spike confirmation on breakouts (2x 20-bar avg volume)

Why this might beat Sharpe=5.4:
- KAMA reduces false signals in ranging markets (adaptive smoothing)
- MACD histogram leads price action better than RSI extremes
- BBW filter avoids trading during volatility compression (squeeze → expansion)
- EMA crossover trend filter works consistently across BTC/ETH/SOL
- Discrete position sizing (0.0, ±0.20, ±0.35) minimizes churn costs
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_bbw_regime_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts smoothing based on market efficiency
    ER (Efficiency Ratio) = |change| / sum(|individual changes|)
    Higher ER = trending market = less smoothing
    Lower ER = choppy market = more smoothing
    """
    n = len(close)
    kama = np.zeros(n)
    
    if n < period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period - 1, n):
        price_change = abs(close[i] - close[i - period + 1])
        volatility = np.sum(np.abs(np.diff(close[i - period + 1:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower) / sma
    
    return upper, lower, band_width, sma


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


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, min_periods=period).mean().values


def calculate_volume_spike(volume, period=20, threshold=2.0):
    """Detect volume spikes above threshold * average"""
    n = len(volume)
    avg_volume = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * avg_volume)
    return spike


def calculate_bbw_percentile(band_width, lookback=50):
    """Calculate BBW percentile to detect squeeze vs expansion regime"""
    n = len(band_width)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = band_width[i - lookback + 1:i + 1]
        rank = np.sum(window <= band_width[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    kama_1h = calculate_kama(close, period=21)
    atr_1h = calculate_atr(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper, bb_lower, bb_width, bb_sma = calculate_bollinger_bands(close, period=20)
    bbw_pct = calculate_bbw_percentile(bb_width, lookback=50)
    volume_spike = calculate_volume_spike(volume, period=20, threshold=2.0)
    
    # 4h trend filter using EMA crossover (resample 1h → 4h)
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
    ema_21_4h = calculate_ema(c_4h, period=21)
    ema_55_4h = calculate_ema(c_4h, period=55)
    
    # 4h trend direction based on EMA crossover
    trend_4h = np.zeros(len(c_4h))
    for i in range(55, len(c_4h)):
        if ema_21_4h[i] > ema_55_4h[i] and ema_21_4h[i-1] <= ema_55_4h[i-1]:
            trend_4h[i:] = 1  # Bullish crossover
        elif ema_21_4h[i] < ema_55_4h[i] and ema_21_4h[i-1] >= ema_55_4h[i-1]:
            trend_4h[i:] = -1  # Bearish crossover
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = min(idx_1h_to_4h[i], len(trend_4h) - 1)
        if idx_4h >= 55:
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # MACD histogram thresholds for momentum entry
    MACD_LONG_THRESHOLD = 0.0   # Histogram crossing above zero
    MACD_SHORT_THRESHOLD = 0.0  # Histogram crossing below zero
    
    # BBW regime filter - avoid trading during squeeze (low volatility)
    BBW_MIN_PERCENTILE = 0.20   # Don't trade if BBW in bottom 20% (squeeze)
    BBW_MAX_PERCENTILE = 0.85   # Reduce size if BBW in top 15% (extended)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 55, 21, 26, 50)  # Wait for all indicators
    
    # Track entry prices and positions for trailing stop logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_price = np.zeros(n)  # Track highest price for long trailing
    lowest_price = np.zeros(n)   # Track lowest price for short trailing
    
    for i in range(first_valid, n):
        if np.isnan(kama_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        macd_histogram = macd_hist[i]
        macd_histogram_prev = macd_hist[i - 1] if i > 0 else 0
        bbw_percentile = bbw_pct[i]
        atr = atr_1h[i]
        price = close[i]
        kama_val = kama_1h[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            continue
        
        # BBW regime filter - avoid squeeze periods
        if bbw_percentile < BBW_MIN_PERCENTILE:
            # Volatility squeeze - stay flat or reduce position
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1] * 0.5  # Reduce position
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Update trailing price trackers
        if i > 0:
            highest_price[i] = max(highest_price[i - 1], price)
            lowest_price[i] = min(lowest_price[i - 1], price)
        else:
            highest_price[i] = price
            lowest_price[i] = price
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            
            if prev_side == 1:  # Long position
                # Trail stop at 2.5*ATR from highest price since entry
                stoploss_price = highest_price[i] - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_price[i] = price
                    lowest_price[i] = price
                    continue
            elif prev_side == -1:  # Short position
                # Trail stop at 2.5*ATR from lowest price since entry
                stoploss_price = lowest_price[i] + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_price[i] = price
                    lowest_price[i] = price
                    continue
        
        # Determine position size based on BBW percentile
        if bbw_percentile > BBW_MAX_PERCENTILE:
            current_size = SIZE_HALF  # Reduced size in extended volatility
        else:
            current_size = SIZE_FULL  # Full size in normal volatility
        
        # KAMA confirmation - price relative to KAMA
        kama_long_confirmed = price > kama_val
        kama_short_confirmed = price < kama_val
        
        if trend == 1:  # 4h uptrend
            # MACD histogram crossing above zero (momentum shift)
            macd_long_signal = macd_histogram > MACD_LONG_THRESHOLD and macd_histogram_prev <= MACD_LONG_THRESHOLD
            
            # KAMA confirmation
            if macd_long_signal and kama_long_confirmed:
                # Volume spike confirmation for breakouts
                if volume_spike[i]:
                    signals[i] = current_size
                else:
                    signals[i] = SIZE_HALF
                position_side[i] = 1
                entry_price[i] = price
                highest_price[i] = price
                lowest_price[i] = price
            elif kama_long_confirmed:
                # Hold existing long or stay flat
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                # KAMA not confirmed - exit long
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # MACD histogram crossing below zero (momentum shift)
            macd_short_signal = macd_histogram < MACD_SHORT_THRESHOLD and macd_histogram_prev >= MACD_SHORT_THRESHOLD
            
            # KAMA confirmation
            if macd_short_signal and kama_short_confirmed:
                # Volume spike confirmation for breakouts
                if volume_spike[i]:
                    signals[i] = -current_size
                else:
                    signals[i] = -SIZE_HALF
                position_side[i] = -1
                entry_price[i] = price
                highest_price[i] = price
                lowest_price[i] = price
            elif kama_short_confirmed:
                # Hold existing short or stay flat
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                # KAMA not confirmed - exit short
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
    
    return signals