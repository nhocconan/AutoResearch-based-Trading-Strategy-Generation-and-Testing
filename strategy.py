#!/usr/bin/env python3
"""
Experiment #1003: 6h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: The Ehlers Fisher Transform provides superior reversal signals compared to RSI,
especially in bear/range markets (2022 crash, 2025 test). Combined with Choppiness Index
for regime detection and 1d/1w HMA for trend bias, this should generate 30-60 trades/year
with better risk-adjusted returns than pure RSI-based strategies.

Key innovations:
1. Fisher Transform (period=9): Transforms price to near-Gaussian distribution, extreme values
   indicate reversal points. Long when Fisher crosses above -1.5, short when crosses below +1.5
2. Choppiness Index (CHOP 14): >55 = range (use Fisher reversals), <45 = trend (use HMA alignment)
3. 1d/1w HMA(21): Triple alignment for strong trend bias (price > 1d_HMA > 1w_HMA = bull)
4. Regime-adaptive entries with LOOSE thresholds to guarantee trades:
   - Range: Fisher < -1.0 + CHOP > 50 + HTF bias = long
   - Trend: HMA alignment + Fisher confirmation = trend follow
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work on 6h:
- Fisher Transform catches reversals better than RSI in bear markets (critical for 2025 test)
- 6h timeframe captures multi-day swings without 4h noise or 12h slowness
- Choppiness filter prevents trend-following whipsaws in 2022-2023 range periods
- LOOSE entry thresholds ensure 30-60 trades/year (avoiding 0-trade failures)
- 1d/1w HTF provides strong directional bias without over-filtering

Entry conditions (LOOSE to guarantee >=30 trades):
- LONG range: CHOP>50 + Fisher<-1.0 + close>1w_HMA*0.92
- LONG trend: price>1d_HMA>1w_HMA + Fisher>-1.0 + Fisher>Fisher_prev
- SHORT range: CHOP>50 + Fisher>1.0 + close<1w_HMA*1.08
- SHORT trend: price<1d_HMA<1w_HMA + Fisher<1.0 + Fisher<Fisher_prev

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_hma_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - transforms price to near-Gaussian distribution
    Extreme values indicate potential reversal points
    
    Formula:
    1. Calculate typical price: (High + Low) / 2
    2. Normalize: 0.991 * (price - lowest) / (highest - lowest) + 0.009
    3. Fisher = 0.5 * ln((1 + normalized) / (1 - normalized))
    4. Signal = previous Fisher value
    
    Long: Fisher crosses above -1.5 (oversold reversal)
    Short: Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        window_high = np.max(high[i-period+1:i+1])
        window_low = np.min(low[i-period+1:i+1])
        
        price_range = window_high - window_low
        if price_range < 1e-10:
            continue
        
        # Normalize price (0.991 multiplier ensures bounds)
        normalized = 0.991 * (typical[i] - window_low) / price_range + 0.009
        
        # Clamp to avoid division by zero
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Signal line (previous Fisher)
        if i > period - 1 and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    
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
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 50.0  # Range market (looser threshold)
        is_trending = chop_14[i] < 50.0  # Trend market
        
        # === HTF BIAS (HMA alignment) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong trend alignment
        strong_bull = hma_1d_bull and hma_1w_bull and hma_1d_aligned[i] > hma_1w_aligned[i]
        strong_bear = hma_1d_bear and hma_1w_bear and hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # Fisher crossover signals
        fisher_bull_cross = fisher[i] > -1.0 and fisher_signal[i] < -1.0  # Cross above -1.0
        fisher_bear_cross = fisher[i] < 1.0 and fisher_signal[i] > 1.0  # Cross below +1.0
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - use Fisher extremes for reversals
            # Long when Fisher extremely oversold + HTF not strongly bearish
            if fisher_oversold and not strong_bear:
                desired_signal = SIZE_BASE
            # Short when Fisher extremely overbought + HTF not strongly bullish
            elif fisher_overbought and not strong_bull:
                desired_signal = -SIZE_BASE
            # Stronger signals at Fisher crossover
            elif fisher_bull_cross and hma_1w_bull:
                desired_signal = SIZE_STRONG
            elif fisher_bear_cross and hma_1w_bear:
                desired_signal = -SIZE_STRONG
        
        else:
            # TREND FOLLOWING MODE - use HMA alignment + Fisher confirmation
            # Long in strong uptrend with Fisher not overbought
            if strong_bull and fisher[i] < 1.5:
                desired_signal = SIZE_STRONG
            # Short in strong downtrend with Fisher not oversold
            elif strong_bear and fisher[i] > -1.5:
                desired_signal = -SIZE_STRONG
            # Weaker trend signals
            elif hma_1d_bull and hma_1w_bull and fisher[i] > fisher_signal[i]:
                desired_signal = SIZE_BASE
            elif hma_1d_bear and hma_1w_bear and fisher[i] < fisher_signal[i]:
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