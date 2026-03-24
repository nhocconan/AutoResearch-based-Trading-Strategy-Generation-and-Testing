#!/usr/bin/env python3
"""
Experiment #711: 6h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: 6h timeframe needs regime-adaptive logic. Fisher Transform catches reversals
better than RSI in bear/range markets (2022 crash, 2025 bear). Choppiness Index filters
whipsaw periods. Different entry logic per regime: trend-follow when CHOP<38.2, mean-revert
when CHOP>61.8. This is DIFFERENT from #703 (pure HMA+RSI) which had Sharpe=-0.044.

Key innovations:
1. Ehlers Fisher Transform (period=9) - superior reversal detection vs RSI
2. Choppiness Index (14) - regime filter (CHOP>61.8 = range, CHOP<38.2 = trend)
3. 1d HMA(21) + 1w HMA(21) - HTF bias confirmation (proven from #703)
4. Regime-adaptive entries:
   - Trending: Fisher crosses -1.5 + HTF bull → long; Fisher crosses +1.5 + HTF bear → short
   - Ranging: Fisher extremes + price at Bollinger bands → mean reversion
5. ATR(14) 2.5x trailing stop - risk management
6. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Entry conditions (designed for 30-60 trades/year on 6h):
- LONG trend: CHOP<50 + 1d/1w HMA bull + Fisher crosses above -1.5
- LONG range: CHOP>61.8 + price<BB_lower + Fisher<-1.0
- SHORT trend: CHOP<50 + 1d/1w HMA bear + Fisher crosses below +1.5
- SHORT range: CHOP>61.8 + price>BB_upper + Fisher>+1.0

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_hma_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals better than RSI in bear/range markets
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Normalize to -1 to +1 range
    highest = pd.Series(typical).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(typical).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    normalized = 0.66 * ((typical - lowest) / range_val - 0.5) + 0.67 * np.roll(normalized, 1)
    normalized[0] = 0.0
    for i in range(1, n):
        if i < period:
            normalized[i] = 0.0
        else:
            normalized[i] = 0.66 * ((typical[i] - lowest[i]) / range_val[i] - 0.5) + 0.67 * normalized[i-1]
    
    # Clamp to avoid log errors
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    
    # Fisher signal (previous bar for no look-ahead)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = np.nan
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # True range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Sum of TR over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high - Lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = hh - ll
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    # Choppiness formula
    chop = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands - for mean reversion entries"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range, CHOP < 38.2 = trend, between = neutral
        is_trending = chop[i] < 45.0  # Slightly relaxed for more trades
        is_ranging = chop[i] > 55.0   # Slightly relaxed for more trades
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 (bullish reversal)
        fisher_long_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        # Fisher crosses below +1.5 (bearish reversal)
        fisher_short_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher at extremes (for range trading)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === PRICE POSITION vs BOLLINGER BANDS ===
        price_at_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
        price_at_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # LONG: Trending regime + HTF bull + Fisher cross
        if is_trending and htf_1d_bull and htf_1w_bull and fisher_long_cross:
            desired_signal = SIZE_STRONG
        # LONG: Ranging regime + price at BB lower + Fisher oversold
        elif is_ranging and price_at_lower and fisher_oversold:
            desired_signal = SIZE_BASE
        # LONG: HTF strong bull + Fisher cross (regime-neutral)
        elif htf_1d_bull and htf_1w_bull and fisher_long_cross:
            desired_signal = SIZE_BASE
        
        # SHORT: Trending regime + HTF bear + Fisher cross
        elif is_trending and htf_1d_bear and htf_1w_bear and fisher_short_cross:
            desired_signal = -SIZE_STRONG
        # SHORT: Ranging regime + price at BB upper + Fisher overbought
        elif is_ranging and price_at_upper and fisher_overbought:
            desired_signal = -SIZE_BASE
        # SHORT: HTF strong bear + Fisher cross (regime-neutral)
        elif htf_1d_bear and htf_1w_bear and fisher_short_cross:
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
                entry_atr = atr[i]
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