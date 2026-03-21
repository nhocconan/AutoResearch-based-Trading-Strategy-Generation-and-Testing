#!/usr/bin/env python3
"""
Experiment #010: 4h Regime-Adaptive Strategy with Daily Trend Filter
Hypothesis: 4h timeframe balances noise reduction with trade frequency.
Use 1d HMA for major trend regime detection (bull/bear).
Use Bollinger BandWidth percentile to detect range vs trend regimes.
In trend regime: follow 4h HMA direction with RSI pullback entries.
In range regime: mean reversion with Z-score extremes.
ATR-based stoploss at 2.5*ATR protects against crashes.
Position sizing: 0.25 base, discrete levels to minimize fee churn.
This should work in both 2021-2024 bull/bear cycles and 2025 range market.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_regime_adaptive_4h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma, std

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_bb_width_percentile(upper, lower, sma, lookback=50):
    """Calculate Bollinger BandWidth percentile for regime detection."""
    bb_width = (upper - lower) / (sma + 1e-10)
    bb_width_series = pd.Series(bb_width)
    percentile = bb_width_series.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: np.searchsorted(np.sort(x.values), x.iloc[-1]) / len(x), raw=False
    ).values
    return np.nan_to_num(percentile, nan=0.5)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    bb_upper, bb_lower, bb_sma, bb_std = calculate_bollinger(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_upper, bb_lower, bb_sma, 50)
    
    # Volume SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=0.0)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(100, n):
        # Daily trend regime (major direction)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 4h trend direction
        hma_trend_long = hma_16[i] > hma_48[i]
        hma_trend_short = hma_16[i] < hma_48[i]
        
        # Regime detection: low BB width = range, high = trend
        is_trend_regime = bb_width_pct[i] > 0.6  # Top 40% of BB width = trending
        is_range_regime = bb_width_pct[i] < 0.4  # Bottom 40% = ranging
        
        # Volume confirmation
        vol_ok = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        new_signal = 0.0
        
        # TREND REGIME: Follow HMA direction with RSI pullback
        if is_trend_regime:
            # Long: daily bullish + 4h HMA up + RSI pullback (not overbought)
            if daily_bullish and hma_trend_long and 35 < rsi[i] < 65 and vol_ok:
                new_signal = SIZE
            # Short: daily bearish + 4h HMA down + RSI bounce (not oversold)
            elif daily_bearish and hma_trend_short and 35 < rsi[i] < 65 and vol_ok:
                new_signal = -SIZE
            # HMA crossover entry
            elif daily_bullish and hma_16[i] > hma_48[i] and hma_16[i-1] <= hma_48[i-1]:
                new_signal = SIZE
            elif daily_bearish and hma_16[i] < hma_48[i] and hma_16[i-1] >= hma_48[i-1]:
                new_signal = -SIZE
        
        # RANGE REGIME: Mean reversion with Z-score
        elif is_range_regime:
            # Long: Z-score oversold + near lower BB
            if zscore[i] < -1.5 and close[i] < bb_lower[i] * 1.02:
                new_signal = SIZE * 0.8  # Smaller size for mean reversion
            # Short: Z-score overbought + near upper BB
            elif zscore[i] > 1.5 and close[i] > bb_upper[i] * 0.98:
                new_signal = -SIZE * 0.8
            # RSI extremes in range
            elif rsi[i] < 30 and close[i] < bb_sma[i]:
                new_signal = SIZE * 0.6
            elif rsi[i] > 70 and close[i] > bb_sma[i]:
                new_signal = -SIZE * 0.6
        
        # NEUTRAL REGIME: Wait for clear signals
        else:
            # Only enter on strong HMA crossover with daily alignment
            if daily_bullish and hma_16[i] > hma_48[i] and hma_16[i-1] <= hma_48[i-1] and rsi[i] > 45:
                new_signal = SIZE
            elif daily_bearish and hma_16[i] < hma_48[i] and hma_16[i-1] >= hma_48[i-1] and rsi[i] < 55:
                new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs - take partial profit at 3R
            elif close[i] > entry_price[i-1] + 3.0 * atr[i]:
                if new_signal == SIZE:
                    new_signal = HALF_SIZE  # Take partial profit
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts - take partial profit at 3R
            elif close[i] < entry_price[i-1] - 3.0 * atr[i]:
                if new_signal == -SIZE:
                    new_signal = -HALF_SIZE  # Take partial profit
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            else:
                entry_price[i] = entry_price[i-1]
                highest_since_entry[i] = max(highest_since_entry[i-1], close[i])
                lowest_since_entry[i] = min(lowest_since_entry[i-1], close[i])
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else close[i]
            lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else close[i]
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals