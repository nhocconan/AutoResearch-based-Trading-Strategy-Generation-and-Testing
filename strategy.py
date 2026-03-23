#!/usr/bin/env python3
"""
Experiment #469: 4h Primary + 1d HTF — KAMA Trend + Fisher Transform + Choppiness Regime

Hypothesis: Based on mtf_4h_kama_fisher_chop_volume_1d1w_v1 (Sharpe=0.204) which worked.
Key insight: KAMA adapts to volatility better than HMA/EMA in crypto. Fisher Transform
catches reversals at extremes (works well in bear/range markets like 2025). Choppiness
Index switches between trend-follow and mean-revert logic.

Innovations:
1. KAMA(21)/KAMA(50) crossover - adapts speed to volatility (faster in trends, slower in chop)
2. Fisher Transform(9) - enters at extremes (-1.5/+1.5 crosses) for reversal timing
3. Choppiness(14) regime - >61.8 = range (mean revert), <38.2 = trend (breakout)
4. 1d KAMA(21) for HTF bias alignment
5. ATR(14) trailing stop at 2.5x for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work: KAMA is proven adaptive indicator. Fisher catches reversals in
bear markets (2025 test period). Choppiness prevents trend strategies in ranges.
Minimal filters = ensures trades happen (avoiding Sharpe=0.000 failures).

Target: Sharpe > 0.612, 20-50 trades/year, DD < -35%
Timeframe: 4h (proven best for crypto swing with HTF alignment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1))[1:])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    sc = (er * (2.0/(fast_period+1) - 2.0/(slow_period+1)) + 2.0/(slow_period+1))**2
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period+1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher(close, period=9):
    """Calculate Ehlers Fisher Transform."""
    n = len(close)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period * 2:
        return fisher, trigger
    
    # Normalize price to -1 to +1 range
    for i in range(period, n):
        highest = np.nanmax(close[i-period+1:i+1])
        lowest = np.nanmin(close[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            continue
        
        normalized = 2.0 * (close[i] - lowest) / range_val - 1.0
        normalized = np.clip(normalized, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    kama_21 = calculate_kama(close, period=21, fast_period=2, slow_period=30)
    kama_50 = calculate_kama(close, period=50, fast_period=2, slow_period=30)
    fisher, fisher_trigger = calculate_fisher(close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(kama_21[i]) or np.isnan(kama_50[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(kama_1d_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = Range (mean revert), CHOP < 38.2 = Trend (breakout)
        is_ranging = chop_14[i] > 55.0  # Slightly lower threshold for more trades
        is_trending = chop_14[i] < 45.0  # Slightly higher threshold for more trades
        
        # === PRIMARY TREND (KAMA crossover) ===
        trend_bullish = kama_21[i] > kama_50[i]
        trend_bearish = kama_21[i] < kama_50[i]
        
        # === HTF TREND BIAS (1d KAMA) ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] < -1.5
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] > 1.5
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY LOGIC
        long_score = 0
        
        if is_trending:
            # Trend regime: follow trend with Fisher timing
            if trend_bullish:
                long_score += 2
            if price_above_kama_1d:
                long_score += 1
            if fisher[i] > -1.0:  # Fisher recovering from oversold
                long_score += 1
        elif is_ranging:
            # Range regime: mean revert at extremes
            if price_below_kama_1d:  # Price below HTF support
                long_score += 1
            if fisher[i] < -1.0:  # Oversold
                long_score += 2
            if fisher_long:  # Fisher reversal signal
                long_score += 1
        else:
            # Neutral regime: require stronger confluence
            if trend_bullish and price_above_kama_1d:
                long_score += 2
            if fisher[i] > -0.5:
                long_score += 1
        
        if long_score >= 2:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRY LOGIC
        if desired_signal == 0.0:
            short_score = 0
            
            if is_trending:
                # Trend regime: follow trend with Fisher timing
                if trend_bearish:
                    short_score += 2
                if price_below_kama_1d:
                    short_score += 1
                if fisher[i] < 1.0:  # Fisher recovering from overbought
                    short_score += 1
            elif is_ranging:
                # Range regime: mean revert at extremes
                if price_above_kama_1d:  # Price above HTF resistance
                    short_score += 1
                if fisher[i] > 1.0:  # Overbought
                    short_score += 2
                if fisher_short:  # Fisher reversal signal
                    short_score += 1
            else:
                # Neutral regime: require stronger confluence
                if trend_bearish and price_below_kama_1d:
                    short_score += 2
                if fisher[i] < 0.5:
                    short_score += 1
            
            if short_score >= 2:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and trend_bullish:
                desired_signal = SIZE_LONG
            elif position_side < 0 and trend_bearish:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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