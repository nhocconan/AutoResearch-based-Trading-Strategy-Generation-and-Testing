#!/usr/bin/env python3
"""
EXPERIMENT #003 - Supertrend + MACD Momentum with Daily Trend Filter (4h Primary, 1d HTF)
==================================================================================================
Hypothesis: Moving to 4h primary timeframe with 1d HTF filter will reduce noise and whipsaws
compared to the 1h/4h combination in #002. Key improvements:

1. SUPERTREND (ATR=10, mult=3): Cleaner trend identification with built-in ATR-based stops
2. MACD (12,26,9) HISTOGRAM: Better momentum timing than RSI pullbacks (captures acceleration)
3. DAILY SMA(50) FILTER: Master trend filter - only trade 4h signals aligning with daily trend
4. VOLATILITY REGIME FILTER: Avoid trading when Bollinger Band Width is in bottom 20% (chop)

Why this should beat #002 (Donchian+RSI+ADX, 1h/4h, Sharpe=0.128):
- 4h timeframe = fewer false signals, cleaner trends than 1h
- 1d HTF = stronger trend filter than 4h (weekly bias captured)
- Supertrend = built-in stop logic, cleaner than Donchian midpoint
- MACD histogram = momentum acceleration, better timing than RSI levels
- Volatility filter = avoids ranging markets (major drawdown source)

Risk Management:
- Position size: 0.20-0.35 (discrete levels)
- Stoploss: Supertrend flip OR 2.5 ATR trailing stop
- Take profit: Reduce to half at 2.5R, trail stop at 1.5R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_macd_daily_mtf_4h_1d_v1"
timeframe = "4h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize final bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    trend = np.zeros(n)
    
    # First valid bar
    final_upper[period - 1] = upper_band[period - 1]
    final_lower[period - 1] = lower_band[period - 1]
    trend[period - 1] = 1 if close[period - 1] > final_upper[period - 1] else -1
    
    for i in range(period, n):
        # Update upper band
        if upper_band[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i - 1]
        
        # Update lower band
        if lower_band[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i - 1]
        
        # Determine trend
        if trend[i - 1] == 1:
            if close[i] < final_lower[i]:
                trend[i] = -1
                final_upper[i] = upper_band[i]  # Reset upper band on flip
            else:
                trend[i] = 1
        else:
            if close[i] > final_upper[i]:
                trend[i] = 1
                final_lower[i] = lower_band[i]  # Reset lower band on flip
            else:
                trend[i] = -1
    
    # Supertrend value (the active stop level)
    supertrend_values = np.where(trend == 1, final_lower, final_upper)
    
    return supertrend_values, trend


def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    MACD indicator
    Returns: macd_line, signal_line, histogram
    """
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    
    macd_line = (ema_fast - ema_slow).values
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """
    Bollinger Bands
    Returns: upper, middle, lower, bandwidth
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    middle = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bandwidth = (upper - lower) / middle
    
    return upper, middle, lower, bandwidth


def calculate_sma(close, period=50):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 4h INDICATORS (ENTRY TIMING) ==========
    atr_4h = calculate_atr(high, low, close, period=14)
    supertrend_4h, supertrend_direction_4h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    macd_line_4h, macd_signal_4h, macd_hist_4h = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper_4h, bb_mid_4h, bb_lower_4h, bb_width_4h = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Calculate BB width percentile for volatility regime filter
    bb_width_percentile = np.zeros(n)
    lookback = 100
    for i in range(lookback, n):
        if bb_width_4h[i] > 0:
            window = bb_width_4h[i-lookback:i+1]
            bb_width_percentile[i] = np.sum(window <= bb_width_4h[i]) / len(window)
    
    # ========== 1d INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        
        # Daily SMA(50) for master trend
        sma_50_1d = calculate_sma(close_1d, period=50)
        
        # Align to 4h timeframe (auto shift for completed bars)
        sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
        
    except Exception:
        sma_50_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.20    # Standard position
    SIZE_HIGH = 0.35    # High conviction (all filters agree)
    
    # Filter thresholds
    BB_WIDTH_LOW_PERCENTILE = 0.20  # Avoid bottom 20% volatility (chop)
    MACD_HIST_THRESHOLD = 0.0       # Histogram crossing above/below zero
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5
    TP_MULT = 2.5
    TRAIL_MULT = 1.5
    
    first_valid = max(150, 100)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_4h[i]) or atr_4h[i] == 0 or np.isnan(supertrend_direction_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        supertrend_dir = supertrend_direction_4h[i]
        macd_hist = macd_hist_4h[i]
        bb_width_pct = bb_width_percentile[i]
        
        # Daily trend filter (master)
        daily_sma = sma_50_1d_aligned[i]
        daily_trend = 0
        if daily_sma > 0 and price > daily_sma:
            daily_trend = 1
        elif daily_sma > 0 and price < daily_sma:
            daily_trend = -1
        
        # Volatility regime filter - avoid chop
        low_volatility = bb_width_pct < BB_WIDTH_LOW_PERCENTILE
        
        # ========== CHECK EXISTING POSITIONS ==========
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Supertrend flip stoploss
            supertrend_stop = supertrend_4h[i]
            if prev_side == 1 and price < supertrend_stop:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            if prev_side == -1 and price > supertrend_stop:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            # ATR stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2.5R) - reduce to half
                tp_price = prev_entry + TP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE  # Reduce to half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1.5R profit
                if prev_tp:
                    trail_stop = current_high - TRAIL_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2.5R) - reduce to half
                tp_price = prev_entry - TP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE  # Reduce to half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1.5R profit
                if prev_tp:
                    trail_stop = current_low + TRAIL_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== NEW ENTRY LOGIC ==========
        # Must have sufficient volatility (not chop)
        if low_volatility:
            signals[i] = 0.0
            continue
        
        # Daily trend must agree with entry direction (cross-asset filter)
        # Long setup: Daily trend up + 4h Supertrend up + MACD histogram positive
        long_setup = (
            daily_trend == 1 and
            supertrend_dir == 1 and
            macd_hist > MACD_HIST_THRESHOLD
        )
        
        # Short setup: Daily trend down + 4h Supertrend down + MACD histogram negative
        short_setup = (
            daily_trend == -1 and
            supertrend_dir == -1 and
            macd_hist < -MACD_HIST_THRESHOLD
        )
        
        # Determine position size
        # High conviction: strong trend alignment + MACD momentum
        high_conviction = (
            daily_trend != 0 and
            supertrend_dir != 0 and
            daily_trend == supertrend_dir and
            abs(macd_hist) > np.nanstd(macd_hist[:i]) if i > 0 else False
        )
        
        if long_setup:
            size = SIZE_HIGH if high_conviction else SIZE_BASE
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_setup:
            size = SIZE_HIGH if high_conviction else SIZE_BASE
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
    
    return signals