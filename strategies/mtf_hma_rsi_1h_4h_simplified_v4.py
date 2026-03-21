#!/usr/bin/env python3
"""
EXPERIMENT #027 - MTF HMA+RSI Simplified (1h+4h v1)
==================================================================================================
Hypothesis: The current best (#022) uses 30m/4h with Sharpe=1.153. Testing same logic on 1h base
timeframe should produce cleaner signals with less noise while maintaining sufficient trade count.
Key differences from current strategy (#004):
- Simpler 2-TF setup (1h+4h) vs complex 3-TF (15m+1h+4h)
- HMA trend filter (proven in #022) vs Supertrend
- Fewer entry conditions (less filter rejection)
- Position size 0.30 (vs 0.35) for better drawdown control
- Stoploss 1.5*ATR (tighter) vs 2.0*ATR

Why this should work:
- 30m/4h proved successful in #022, 1h should be similar but cleaner
- Fewer timeframe alignments = less complexity = fewer bugs
- HMA is faster than EMA, catches trends earlier
- RSI pullback entries work well in trending markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_1h_4h_simplified_v4"
timeframe = "1h"
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h_fast = calculate_hma(close, period=16)
    hma_1h_slow = calculate_hma(close, period=48)
    
    # Get 4h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h indicators
        hma_4h_fast = calculate_hma(c_4h, period=16)
        hma_4h_slow = calculate_hma(c_4h, period=48)
        rsi_4h = calculate_rsi(c_4h, period=14)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        hma_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_fast)
        hma_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slow)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_fast_aligned = np.zeros(n)
        hma_4h_slow_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n) + 50
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # ATR stoploss multiplier (tighter for 1h timeframe)
    ATR_STOP_MULT = 1.5
    
    first_valid = max(100, 48, 14 * 2)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get aligned MTF values
        hma_4h_f = hma_4h_fast_aligned[i] if i < len(hma_4h_fast_aligned) else 0
        hma_4h_s = hma_4h_slow_aligned[i] if i < len(hma_4h_slow_aligned) else 0
        rsi_4h_val = rsi_4h_aligned[i] if i < len(rsi_4h_aligned) else 50
        
        # 4h trend filter (HMA fast > slow = bullish, fast < slow = bearish)
        trend_4h = 0
        if hma_4h_f > hma_4h_s and hma_4h_f > 0:
            trend_4h = 1
        elif hma_4h_f < hma_4h_s and hma_4h_f > 0:
            trend_4h = -1
        
        # 1h trend filter (same HMA crossover)
        trend_1h = 0
        if hma_1h_fast[i] > hma_1h_slow[i] and hma_1h_fast[i] > 0:
            trend_1h = 1
        elif hma_1h_fast[i] < hma_1h_slow[i] and hma_1h_fast[i] > 0:
            trend_1h = -1
        
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
            
            # Stoploss check (1.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
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
        
        # Entry logic: 4h trend + 1h trend agreement + 1h RSI pullback
        price = close[i]
        
        # Both 4h and 1h must agree on trend direction
        if trend_4h == 1 and trend_1h == 1:  # Bullish trend on both
            # RSI pullback on 1h (not overbought, in healthy range)
            if RSI_LONG_MIN <= rsi_1h[i] <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1 and trend_1h == -1:  # Bearish trend on both
            # RSI pullback on 1h (not oversold, in healthy range)
            if RSI_SHORT_MIN <= rsi_1h[i] <= RSI_SHORT_MAX:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals