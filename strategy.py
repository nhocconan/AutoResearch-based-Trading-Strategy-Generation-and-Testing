#!/usr/bin/env python3
"""
EXPERIMENT #005 - MTF HMA+BB+RSI+ZSCORE (15m+1h+4h v1)
==================================================================================================
Hypothesis: Combine 4h HMA trend (proven in current best) + 1h Bollinger Band position 
(mean reversion within trend) + 15m RSI pullback entry + Z-score regime filter.

Key differences from current best (mtf_hma_rsi_zscore_v1):
- Add Bollinger Band position filter on 1h (price %B between 0.3-0.7 for pullback entries)
- Use 15m base timeframe instead of 1h (more entry opportunities)
- Z-score on 1h instead of 15m (smoother regime detection)
- Three-timeframe confirmation reduces false signals

Why this should work:
- 4h HMA provides strong trend direction (proven in baseline)
- 1h BB %B ensures we enter on pullbacks within the trend (not chasing)
- 15m RSI gives precise entry timing
- Z-score filter avoids extreme regimes
- Conservative position sizing (0.25-0.35) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_bb_rsi_zscore_15m_1h_4h_v1"
timeframe = "15m"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and %B"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    percent_b = np.zeros(n)
    for i in range(n):
        if upper[i] != lower[i]:
            percent_b[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
        else:
            percent_b[i] = 0.5
    
    return upper, middle, lower, percent_b


def calculate_zscore(close, period=20):
    """Calculate Z-score"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(n):
        if std[i] > 0:
            zscore[i] = (close[i] - mean[i]) / std[i]
        else:
            zscore[i] = 0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    
    # Get 1h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_1h = get_htf_data(prices, '1h')
        c_1h = df_1h['close'].values
        h_1h = df_1h['high'].values
        l_1h = df_1h['low'].values
        
        # 1h Bollinger Bands for pullback detection
        _, _, _, bb_pct_1h = calculate_bollinger_bands(c_1h, period=20, std_mult=2.0)
        
        # 1h Z-score for regime filter
        zscore_1h = calculate_zscore(c_1h, period=20)
        
        # Align 1h indicators to 15m timeframe (auto shift for completed bars)
        bb_pct_1h_aligned = align_htf_to_ltf(prices, df_1h, bb_pct_1h)
        zscore_1h_aligned = align_htf_to_ltf(prices, df_1h, zscore_1h)
    except Exception:
        # Fallback if mtf_data fails
        bb_pct_1h_aligned = np.full(n, 0.5)
        zscore_1h_aligned = np.zeros(n)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(c_4h, period=21)
        
        # Align 4h indicators to 15m timeframe
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        c_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
        
        # Calculate 4h trend direction (price vs HMA)
        trend_4h = np.zeros(n)
        for i in range(n):
            if i < len(c_4h_aligned) and i < len(hma_4h_aligned):
                if c_4h_aligned[i] > hma_4h_aligned[i]:
                    trend_4h[i] = 1
                elif c_4h_aligned[i] < hma_4h_aligned[i]:
                    trend_4h[i] = -1
    except Exception:
        hma_4h_aligned = np.zeros(n)
        c_4h_aligned = np.zeros(n)
        trend_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # BB %B thresholds for pullback within trend
    BB_LONG_MIN = 0.30
    BB_LONG_MAX = 0.55
    BB_SHORT_MIN = 0.45
    BB_SHORT_MAX = 0.70
    
    # Z-score regime filter (avoid extremes)
    ZSCORE_MAX = 2.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 14 * 2, 20, 26 + 9)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get aligned MTF values
        trend_4h_val = trend_4h[i] if i < len(trend_4h) else 0
        bb_pct_1h = bb_pct_1h_aligned[i] if i < len(bb_pct_1h_aligned) else 0.5
        zscore_1h = zscore_1h_aligned[i] if i < len(zscore_1h_aligned) else 0
        
        # Z-score regime filter - avoid extreme moves
        if abs(zscore_1h) > ZSCORE_MAX:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Hold position
                signals[i] = signals[i - 1]
                position_side[i] = prev_side
                entry_price[i] = prev_entry
                tp_triggered[i] = prev_tp
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
                continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Hold position
                signals[i] = signals[i - 1]
                position_side[i] = prev_side
                entry_price[i] = prev_entry
                tp_triggered[i] = prev_tp
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
                continue
        
        # Exit position if trend changes
        if trend_4h_val == 0 or (position_side[i - 1] == 1 and trend_4h_val == -1) or (position_side[i - 1] == -1 and trend_4h_val == 1):
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Entry logic: 4h trend + 1h BB pullback + 15m RSI timing
        price = close[i]
        
        if trend_4h_val == 1:  # Bullish trend on 4h
            # BB %B in pullback zone on 1h (0.30-0.55)
            # RSI pullback on 15m (35-55)
            if (BB_LONG_MIN <= bb_pct_1h <= BB_LONG_MAX and 
                RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend_4h_val == -1:  # Bearish trend on 4h
            # BB %B in pullback zone on 1h (0.45-0.70)
            # RSI pullback on 15m (45-65)
            if (BB_SHORT_MIN <= bb_pct_1h <= BB_SHORT_MAX and 
                RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX):
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals