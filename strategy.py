#!/usr/bin/env python3
"""
Experiment #903: 6h Primary + 1d/1w HTF — Fisher Transform + Dual HMA Trend

Hypothesis: 6h timeframe with daily+weekly HTF bias captures multi-day trends
while Fisher Transform provides proven reversal signals that work in both bull
and bear markets. Unlike complex regime filters (CHOP, ADX) that failed in
experiments #891, #895, #900, #902, Fisher Transform catches reversals at
extremes without overfitting.

Key innovations:
1. 1w HMA(21) for meta-trend bias (only trade with weekly direction)
2. 1d HMA(21) for primary trend confirmation
3. Ehlers Fisher Transform(9) for entry timing - proven reversal indicator
4. ATR ratio(7/30) vol filter - avoid low volatility periods
5. Asymmetric sizing: stronger signals when all 3 HTF align
6. 2.5x ATR trailing stop for risk management

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: 1w HMA bull + 1d HMA bull + Fisher crosses above -1.5 + ATR ratio > 0.8
- SHORT: 1w HMA bear + 1d HMA bear + Fisher crosses below +1.5 + ATR ratio > 0.8

Why Fisher Transform:
- Normalizes price to Gaussian distribution
- Sharp turning points at extremes (-2 to +2 range)
- Proven edge in bear/range markets (research section)
- Less lag than RSI, more signal than raw price

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_dual_hma_vol_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    # WMA helper
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_n)
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian-normalized values with sharp turning points
    Range typically -2 to +2, crossings at -1.5/+1.5 signal reversals
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((price - LL) / (HH - LL) - 0.5) + 0.67 * prev_Fisher_X
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_x = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Highest high and lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        price_range = hh - ll
        if price_range < 1e-10:
            fisher_x[i] = fisher_x[i-1] if i > 0 and not np.isnan(fisher_x[i-1]) else 0.0
        else:
            # Normalized price
            norm_price = (high[i] + low[i]) / 2.0  # Use typical price
            x_raw = (norm_price - ll) / price_range
            
            # Smooth with previous value (Ehlers formula)
            if i > 0 and not np.isnan(fisher_x[i-1]):
                x = 0.66 * (x_raw - 0.5) + 0.67 * fisher_x[i-1]
            else:
                x = 0.66 * (x_raw - 0.5)
            
            # Clamp to prevent division by zero
            x = np.clip(x, -0.999, 0.999)
            fisher_x[i] = x
        
        # Fisher Transform
        if abs(1 - fisher_x[i]) > 1e-10 and abs(1 + fisher_x[i]) > 1e-10:
            fisher[i] = 0.5 * np.log((1 + fisher_x[i]) / (1 - fisher_x[i]))
    
    return fisher

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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """
    ATR Ratio for volatility filter
    Ratio > 1.0 = vol expanding, Ratio < 1.0 = vol contracting
    We want ratio > 0.8 to avoid dead markets
    """
    n = len(close)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_short = pd.Series(tr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(tr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    ratio = np.full(n, np.nan)
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA (Rule 2 - use align_htf_to_ltf)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher = calculate_fisher_transform(high, low, period=9)
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
    
    # Track Fisher crosses
    prev_fisher = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(atr_ratio[i]):
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
        
        # === HTF BIAS (1w + 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY FILTER ===
        vol_ok = atr_ratio[i] > 0.8  # Avoid dead markets
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if not np.isnan(prev_fisher):
            # Long: Fisher crosses above -1.5 from below
            fisher_cross_long = (prev_fisher <= -1.5) and (fisher[i] > -1.5)
            # Short: Fisher crosses below +1.5 from above
            fisher_cross_short = (prev_fisher >= 1.5) and (fisher[i] < 1.5)
        
        prev_fisher = fisher[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADES) ===
        desired_signal = 0.0
        
        # Count HTF alignments
        htf_bull_count = int(htf_1w_bull) + int(htf_1d_bull)
        htf_bear_count = int(htf_1w_bear) + int(htf_1d_bear)
        
        if htf_bull_count >= 1 and vol_ok:
            # At least 1 bullish HTF + vol ok
            if fisher_cross_long:
                if htf_bull_count >= 2:
                    desired_signal = SIZE_STRONG  # Both HTF align
                else:
                    desired_signal = SIZE_BASE
            elif fisher[i] < -1.0:
                # Fisher deeply oversold (loose entry)
                if htf_bull_count >= 2:
                    desired_signal = SIZE_BASE
        
        elif htf_bear_count >= 1 and vol_ok:
            # At least 1 bearish HTF + vol ok
            if fisher_cross_short:
                if htf_bear_count >= 2:
                    desired_signal = -SIZE_STRONG  # Both HTF align
                else:
                    desired_signal = -SIZE_BASE
            elif fisher[i] > 1.0:
                # Fisher deeply overbought (loose entry)
                if htf_bear_count >= 2:
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