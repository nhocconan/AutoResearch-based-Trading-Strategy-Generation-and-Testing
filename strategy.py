#!/usr/bin/env python3
"""
EXPERIMENT #003 - MTF HMA+RSI+ATR (1h+4h v1)
==================================================================================================
Hypothesis: Use 1h primary timeframe with 4h trend filter. HMA provides faster trend detection
than EMA with less lag than SMA. RSI pullback entries in direction of 4h trend. ATR-based
stoploss ensures risk control. Simpler than previous attempt to avoid crashes.

Why this should work:
- 1h has good balance of signal frequency vs noise (more trades than 4h, cleaner than 15m)
- 4h trend filter reduces whipsaws (proven in best strategy mtf_hma_rsi_zscore_v1)
- HMA(16/48) crossover is proven trend indicator
- RSI(14) pullback to 40-60 range for entries
- ATR(14) stoploss at 2.5x for risk control
- Position size 0.30 (discrete, within 0.20-0.35 range)
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_atr_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


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


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.ones(n) * 100
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_fast_1h = calculate_hma(close, period=16)
    hma_slow_1h = calculate_hma(close, period=48)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        from mtf_data import get_htf_data, align_htf_to_ltf
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        hma_fast_4h = calculate_hma(c_4h, period=16)
        hma_slow_4h = calculate_hma(c_4h, period=48)
        hma_fast_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_fast_4h)
        hma_slow_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_slow_4h)
    except Exception:
        # Fallback if mtf_data not available
        hma_fast_4h_aligned = hma_fast_1h
        hma_slow_4h_aligned = hma_slow_1h
    
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
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # Minimum bars for valid signals
    first_valid = max(100, 48, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        current_price = close[i]
        
        # Skip if invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 0:
            signals[i] = 0.0
            if i > 0:
                position_side[i] = position_side[i - 1]
            continue
        
        # Get 4h trend (use previous aligned value to avoid look-ahead)
        idx_4h = max(0, i // 4 - 1)  # 4 x 1h = 4h, minus 1 for completed bar
        idx_4h = min(idx_4h, len(hma_fast_4h_aligned) - 1)
        
        hma_fast_4h_val = hma_fast_4h_aligned[idx_4h]
        hma_slow_4h_val = hma_slow_4h_aligned[idx_4h]
        
        # Determine 4h trend direction (HMA crossover)
        trend_4h = 0
        if hma_fast_4h_val > hma_slow_4h_val and current_price > hma_fast_4h_val:
            trend_4h = 1
        elif hma_fast_4h_val < hma_slow_4h_val and current_price < hma_fast_4h_val:
            trend_4h = -1
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, current_price)
                current_low = prev_low if prev_low > 0 else current_price
            else:
                current_high = prev_high if prev_high > 0 else current_price
                current_low = min(prev_low, current_price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if current_price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and current_price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if current_price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if current_price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and current_price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
                    if current_price > trail_stop:
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
        
        # Entry logic: 4h trend + 1h RSI pullback + HMA confirmation
        rsi_val = rsi_1h[i]
        hma_fast_val = hma_fast_1h[i]
        hma_slow_val = hma_slow_1h[i]
        
        # LONG entry: 4h uptrend + 1h RSI pullback + 1h HMA bullish
        if trend_4h == 1 and RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
            if hma_fast_val > hma_slow_val and current_price > hma_fast_val:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = current_price
                tp_triggered[i] = False
                highest_since_entry[i] = current_price
                lowest_since_entry[i] = current_price
                continue
        
        # SHORT entry: 4h downtrend + 1h RSI pullback + 1h HMA bearish
        if trend_4h == -1 and RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
            if hma_fast_val < hma_slow_val and current_price < hma_fast_val:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = current_price
                tp_triggered[i] = False
                highest_since_entry[i] = current_price
                lowest_since_entry[i] = current_price
                continue
        
        # No entry signal
        signals[i] = 0.0
        position_side[i] = 0
    
    return signals