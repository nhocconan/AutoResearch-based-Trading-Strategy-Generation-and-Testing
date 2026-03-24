#!/usr/bin/env python3
"""
Experiment #030: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Trend + Volatility Breakout

Hypothesis: Previous 1h strategies (#020, #025) failed with 0 trades due to overly strict confluence.
CRSI<15/>85 is too rare. This uses Ehlers Fisher Transform with relaxed thresholds (-1.0/+1.0)
that generate more frequent reversal signals while maintaining quality through HTF confirmation.

Key improvements over failed 1h attempts:
1. Fisher Transform (period=9) - more signals than CRSI extremes, proven reversal catcher
2. 4h HMA(21) - simple trend filter (price above/below), not strict slope requirements
3. 12h ADX(14) - regime filter but relaxed threshold (>18 not >25)
4. NO session/volume filters - these killed trade frequency in #020/#025
5. Relaxed Fisher thresholds: crosses -1.0/+1.0 (not extreme -1.5/+1.5)

Entry Logic:
- Long: Fisher crosses above -1.0 + price>4h_HMA + 12h_ADX>18
- Short: Fisher crosses below +1.0 + price<4h_HMA + 12h_ADX>18
- Alternative: Fisher extreme (<-1.5 or >+1.5) overrides HTF filters for strong reversals

Sizing: 0.25 base, 0.35 with full HTF alignment
Stoploss: 2.5x ATR trailing
Target: 40-70 trades/year, Sharpe>0.4, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_adx_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    Reference: Ehlers, J.F. (2002) "Fisher Transform"
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    for i in range(period, n):
        # Use typical price (H+L)/2
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over lookback period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            if i > period:
                fisher_signal[i] = fisher[i-1]
            else:
                fisher_signal[i] = 0.0
            continue
        
        # Normalize price to 0-1 range
        normalized = (hl2 - lowest) / (highest - lowest)
        
        # Clamp to avoid log(0) or log(inf) issues
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher transform formula
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Signal line is previous fisher value
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # Handle NaN in WMA calculations
    raw_hma = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(raw_hma, sqrt_period)
    return hma

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = strong trend, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's EMA (span = period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di_raw = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di_raw = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI values
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * plus_di_raw[i] / atr[i]
            minus_di[i] = 100.0 * minus_di_raw[i] / atr[i]
    
    # Calculate DX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss and volatility measurement"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    # This reads Parquet files - calling inside loop = 45K file reads = hang
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h ADX for regime filter
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    FULL_SIZE = 0.35
    MAX_SIZE = 0.40
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME FILTER (12h ADX) - relaxed threshold for trade frequency ===
        has_trend = adx_12h_aligned[i] > 18.0
        
        # === TREND DIRECTION (4h HMA) ===
        price_above_hma = close[i] > hma_4h_aligned[i]
        price_below_hma = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.0 (from below) - relaxed from -1.5
        fisher_long_cross = fisher_signal[i] < -1.0 and fisher[i] >= -1.0
        
        # Short: Fisher crosses below +1.0 (from above) - relaxed from +1.5
        fisher_short_cross = fisher_signal[i] > 1.0 and fisher[i] <= 1.0
        
        # === EXTREME FISHER (overrides HTF filters for strong reversals) ===
        fisher_extreme_long = fisher[i] < -1.5
        fisher_extreme_short = fisher[i] > 1.5
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry logic
        if fisher_long_cross or fisher_extreme_long:
            if fisher_extreme_long:
                # Extreme oversold - enter regardless of HTF (strong reversal signal)
                desired_signal = BASE_SIZE
            elif price_above_hma and has_trend:
                # All conditions aligned - full size
                desired_signal = FULL_SIZE
            elif price_above_hma or has_trend:
                # Partial alignment - base size
                desired_signal = BASE_SIZE
        
        # Short entry logic
        elif fisher_short_cross or fisher_extreme_short:
            if fisher_extreme_short:
                # Extreme overbought - enter regardless of HTF
                desired_signal = -BASE_SIZE
            elif price_below_hma and has_trend:
                # All conditions aligned - full size
                desired_signal = -FULL_SIZE
            elif price_below_hma or has_trend:
                # Partial alignment - base size
                desired_signal = -BASE_SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        # Use discrete levels to minimize fee churn (Rule 4)
        if desired_signal >= FULL_SIZE * 0.85:
            final_signal = FULL_SIZE
        elif desired_signal <= -FULL_SIZE * 0.85:
            final_signal = -FULL_SIZE
        elif desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # Clip to max size
        final_signal = np.clip(final_signal, -MAX_SIZE, MAX_SIZE)
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Position flip
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals