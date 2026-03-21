#!/usr/bin/env python3
"""
EXPERIMENT #005 - KAMA Adaptive Trend + Volume + HTF Filter (12h primary, 1d HTF)
================================================================================
Hypothesis: 12h timeframe captures medium-term trends with less noise than 4h.
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - fast in trends,
slow in chop. Combined with 1d HMA(50) for major trend alignment and volume
confirmation, this should produce fewer but higher-quality trades than previous
supertrend/donchian attempts. Bollinger regime filter avoids trading in compression.

Key improvements over failed strategies:
- 12h primary TF (less noise than 4h, more signals than 1d)
- KAMA adapts to volatility (better than static EMA/HMA in chop)
- Volume confirmation filter (avoid low-liquidity false breakouts)
- Stricter trend alignment (both 12h KAMA slope + 1d HMA position)
- Conservative position sizing: 0.25 max (vs 0.30+ in failed strategies)
- 2.5*ATR stoploss (tighter than 3*ATR to cut losses faster)

Indicators:
- Primary TF (12h): KAMA(10,2,30), ATR(14), Bollinger Bands(20,2)
- HTF (1d): HMA(50) for major trend direction
- Volume: 20-period SMA ratio for confirmation
- Regime: BB Width percentile > 40th (avoid extreme compression)

Position sizing: 0.25 base, discrete levels (0.0, ±0.25)
Stoploss: 2.5*ATR trailing
Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_volume_htf_12h_1d_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close, fast_period=10, slow_period=30, smoothing_period=2):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    Adapts to market efficiency ratio - fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(slow_period - 1, n):
        price_change = abs(close[i] - close[i - slow_period + 1])
        volatility = np.sum(np.abs(np.diff(close[i - slow_period + 1:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[slow_period - 1] = close[slow_period - 1]
    
    for i in range(slow_period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower) / sma
    return upper.values, lower.values, band_width.values


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average"""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean()
    volume_ratio = vol_s / (vol_sma + 1e-10)
    return volume_ratio.values


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(series[i]):
            window_data = series[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= series[i]) / len(window_data)
    
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    kama = calculate_kama(close, fast_period=10, slow_period=30, smoothing_period=2)
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2)
    volume_ratio = calculate_volume_ratio(volume, 20)
    
    # Calculate Bollinger Band Width percentile rank (regime filter)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    # Calculate KAMA slope (rate of change over 5 bars)
    kama_slope = np.zeros(n)
    kama_slope[:] = np.nan
    for i in range(5, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i - 5]):
            kama_slope[i] = (kama[i] - kama[i - 5]) / (kama[i - 5] + 1e-10)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - conservative)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    bars_since_entry = 0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(atr[i]) or np.isnan(bb_width_pr[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(kama_slope[i]) or
            atr[i] == 0 or kama[i] == 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter (HTF) - price above/below 1d HMA(50)
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # 12h KAMA trend - price above/below KAMA
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # KAMA slope confirmation (must be positive for long, negative for short)
        slope_threshold = 0.001  # 0.1% slope over 5 bars
        slope_bullish = kama_slope[i] > slope_threshold
        slope_bearish = kama_slope[i] < -slope_threshold
        
        # Regime filter: only trade when BB Width is above 40th percentile
        regime_valid = bb_width_pr[i] > 0.40
        
        # Volume confirmation: volume must be above average (ratio > 0.9)
        volume_confirmed = volume_ratio[i] > 0.9
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: All filters align bullish
        if (kama_trend == 1 and daily_trend == 1 and slope_bullish and 
            regime_valid and volume_confirmed):
            target_signal = SIZE
        
        # Short entry: All filters align bearish
        elif (kama_trend == -1 and daily_trend == -1 and slope_bearish and 
              regime_valid and volume_confirmed):
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            bars_since_entry += 1
            
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
            bars_since_entry = 0
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry - require minimum 5 bars since last exit to reduce churn
                if bars_since_entry >= 5 or bars_since_entry == 0:
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    profit_target_hit = False
                    bars_since_entry = 1
                else:
                    signals[i] = 0.0
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and kama_trend == -1:
                    # Trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                    bars_since_entry = 0
                elif position_side == -1 and kama_trend == 1:
                    # Trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                    bars_since_entry = 0
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals