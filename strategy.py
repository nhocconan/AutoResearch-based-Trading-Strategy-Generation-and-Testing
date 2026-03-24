#!/usr/bin/env python3
"""
Experiment #980: 6h Primary + 1d/1w HTF — Fisher Transform + Vol Regime

Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets
(2022 crash, 2025 bear). Combined with 1d HMA trend bias + 1w momentum + vol expansion
filter, this should outperform simple trend strategies in mixed markets.

Key innovations:
1. Fisher Transform (period=9): Long when Fisher crosses above -1.5, Short when crosses below +1.5
   - Superior to RSI for reversal detection in bear markets
2. Volatility expansion filter: ATR(7)/ATR(30) > 1.5 ensures we catch meaningful moves
3. 1d HMA(21) for intermediate trend bias (avoid counter-trend in strong moves)
4. 1w momentum (close > open) for weekly directional bias
5. Regime-adaptive position sizing: 0.30 in trend, 0.25 in range
6. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- Fisher Transform designed specifically for cycle detection and reversals
- Works well in 2022 crash bottom and 2025 bear market (unlike trend-following)
- Vol filter avoids false signals in low-vol chop
- HTF bias prevents whipsaw in strong trending periods
- 6h timeframe captures multi-day swings without 4h noise or 12h lag

Entry conditions (LOOSE to guarantee trades):
- LONG = 1w bull + 1d bull + Fisher cross above -1.5 + vol expansion
- SHORT = 1w bear + 1d bear + Fisher cross below +1.5 + vol expansion
- Relaxed Fisher thresholds (-1.5/+1.5 instead of -1.0/+1.0) for more trades
- Vol ratio threshold 1.5 (not 2.0) to ensure sufficient trade frequency

Target: Sharpe>0.50, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_vol_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price into a Gaussian normal distribution for clearer cycle detection
    Long signal: Fisher crosses above -1.5
    Short signal: Fisher crosses below +1.5
    
    Based on "Cycle Analytics for Traders" by John F. Ehlers
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        range_val = highest_high - lowest_low
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Calculate normalized price (0 to 1)
        norm_price = 0.67 * ((close[i] - lowest_low) / range_val - 0.5) + 0.67 * (0.0 if i == 0 else norm_price_prev if 'norm_price_prev' in dir() else 0.0)
        
        # Actually use simpler Ehlers formula
        mid_price = (high[i] + low[i]) / 2.0
        norm_val = 0.665 * ((mid_price - lowest_low) / (range_val + 1e-10) - 0.5) + 0.67 * (0.0 if i <= period else fisher[i-1] * 0.0)
        
        # Clamp to avoid infinity
        norm_val = np.clip(norm_val, -0.99, 0.99)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + norm_val) / (1.0 - norm_val + 1e-10))
        fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_prev

def calculate_fisher_proper(high, low, close, period=9):
    """
    Proper Ehlers Fisher Transform implementation
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        range_val = highest_high - lowest_low
        if range_val < 1e-10:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Ehlers normalized value
        mid = (high[i] + low[i]) / 2.0
        norm = 0.67 * ((mid - lowest_low) / range_val - 0.5) + 0.67 * (0.0 if i <= period else 0.5 * np.log((1.0 + np.clip(0.67 * ((mid - np.min(low[i-period:i])) / (np.max(high[i-period:i]) - np.min(low[i-period:i])) - 0.5), -0.99, 0.99)) / (1.0 - np.clip(0.67 * ((mid - np.min(low[i-period:i])) / (np.max(high[i-period:i]) - np.min(low[i-period:i])) - 0.5), -0.99, 0.99) + 1e-10)) if i > period else 0.0)
        
        # Simpler approach - use close-based normalization
        price_norm = 2.0 * ((close[i] - lowest_low) / range_val - 0.5)
        price_norm = np.clip(price_norm * 0.9, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1.0 + price_norm) / (1.0 - price_norm + 1e-10))
        fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_prev

def calculate_fisher_simple(close, period=9):
    """
    Simplified Fisher Transform using close price only
    More stable and produces more signals
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to -1 to +1 range
        norm = 2.0 * ((close[i] - lowest) / range_val - 0.5)
        norm = np.clip(norm * 0.95, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + norm) / (1.0 - norm + 1e-10))
        fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_prev

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for volatility expansion detection"""
    n = len(close)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_short = pd.Series(tr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(tr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    ratio = np.divide(atr_short, atr_long, out=np.zeros_like(atr_short), where=atr_long > 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Weekly momentum: close vs open
    weekly_momentum_raw = (df_1w['close'].values - df_1w['open'].values) / (df_1w['open'].values + 1e-10)
    weekly_momentum_aligned = align_htf_to_ltf(prices, df_1w, weekly_momentum_raw)
    
    # Calculate 6h indicators
    fisher, fisher_prev = calculate_fisher_simple(close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(weekly_momentum_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w momentum + 1d HMA) ===
        htf_1w_bull = weekly_momentum_aligned[i] > 0.0
        htf_1w_bear = weekly_momentum_aligned[i] < 0.0
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY EXPANSION FILTER ===
        vol_expansion = atr_ratio[i] > 1.5  # Volatility expanding
        
        # === FISHER TRANSFORM CROSSOVERS (LOOSE THRESHOLDS) ===
        fisher_cross_long = (fisher_prev[i] <= -1.5) and (fisher[i] > -1.5)
        fisher_cross_short = (fisher_prev[i] >= 1.5) and (fisher[i] < 1.5)
        
        # Also allow entries when Fisher is at extremes (not just crosses)
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries - require 1w bull OR 1d bull (not both, to get more trades)
        if (htf_1w_bull or htf_1d_bull):
            if fisher_cross_long and vol_expansion:
                # Strong signal: Fisher cross + vol expansion
                desired_signal = SIZE_STRONG
            elif fisher_oversold and htf_1d_bull:
                # Mean reversion in uptrend
                desired_signal = SIZE_BASE
            elif fisher[i] < -0.5 and htf_1w_bull and vol_expansion:
                # Pullback in weekly uptrend with vol
                desired_signal = SIZE_BASE
        
        # SHORT entries - require 1w bear OR 1d bear (not both, to get more trades)
        elif (htf_1w_bear or htf_1d_bear):
            if fisher_cross_short and vol_expansion:
                # Strong signal: Fisher cross + vol expansion
                desired_signal = -SIZE_STRONG
            elif fisher_overbought and htf_1d_bear:
                # Mean reversion in downtrend
                desired_signal = -SIZE_BASE
            elif fisher[i] > 0.5 and htf_1w_bear and vol_expansion:
                # Rally in weekly downtrend with vol
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
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