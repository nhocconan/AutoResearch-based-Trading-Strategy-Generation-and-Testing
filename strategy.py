#!/usr/bin/env python3
"""
EXPERIMENT #014 - MTF SUPERTREND+DEMA+MACD+ZSCORE (1h+4h v1)
==================================================================================================
Hypothesis: 4h Supertrend trend + 1h DEMA crossover entry + MACD momentum + Z-score filter
will beat #012 (Sharpe=0.478).

Key changes from #012/#013:
- Trend: 4h Supertrend(ATR=10, mult=3) instead of HMA/EMA - clearer trend signals
- Entry: 1h DEMA(8/21) crossover instead of KAMA - faster response to momentum shifts
- Momentum: 1h MACD histogram confirmation - avoids false breakouts
- Filter: Z-score(20) < 2.0 to avoid extreme entries
- Position size: 0.30 (conservative, discrete levels)
- Stoploss: 2.0*ATR trailing stop

Why this should beat #012:
- Supertrend provides clearer trend direction than HMA crossover (less whipsaw)
- DEMA has less lag than EMA for entry timing
- MACD histogram adds momentum confirmation (reduces false entries)
- Based on proven multi-timeframe approach from #005, #012
- Fixed syntax errors from #013
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_dema_macd_zscore_1h_4h_v1"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period - 1, n):
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
    
    # Initialize
    supertrend[period - 1] = upper_band[period - 1]
    direction[period - 1] = -1
    
    for i in range(period, n):
        if close[i - 1] <= supertrend[i - 1]:
            # Previously bearish
            if close[i] > upper_band[i]:
                # Flip to bullish
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = min(upper_band[i], supertrend[i - 1])
                direction[i] = -1
        else:
            # Previously bullish
            if close[i] < lower_band[i]:
                # Flip to bearish
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = max(lower_band[i], supertrend[i - 1])
                direction[i] = 1
    
    return supertrend, direction


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    # First EMA
    ema1 = np.zeros(n)
    ema1[period - 1] = np.mean(close[:period])
    for i in range(period, n):
        ema1[i] = close[i] * (2 / (period + 1)) + ema1[i - 1] * (1 - 2 / (period + 1))
    
    # Second EMA of EMA1
    ema2 = np.zeros(n)
    ema2[period - 1] = np.mean(ema1[:period])
    for i in range(period, n):
        ema2[i] = ema1[i] * (2 / (period + 1)) + ema2[i - 1] * (1 - 2 / (period + 1))
    
    # DEMA = 2*EMA1 - EMA2
    dema = 2 * ema1 - ema2
    
    return dema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    n = len(close)
    if n < slow:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # Fast EMA
    ema_fast = np.zeros(n)
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = close[i] * (2 / (fast + 1)) + ema_fast[i - 1] * (1 - 2 / (fast + 1))
    
    # Slow EMA
    ema_slow = np.zeros(n)
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = close[i] * (2 / (slow + 1)) + ema_slow[i - 1] * (1 - 2 / (slow + 1))
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = np.zeros(n)
    valid_start = slow + signal - 1
    if valid_start < n:
        signal_line[valid_start] = np.mean(macd_line[slow:valid_start + 1])
        for i in range(valid_start + 1, n):
            signal_line[i] = macd_line[i] * (2 / (signal + 1)) + signal_line[i - 1] * (1 - 2 / (signal + 1))
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    dema8_1h = calculate_dema(close, period=8)
    dema21_1h = calculate_dema(close, period=21)
    _, _, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Resample to 4h for trend filters using actual timestamps
    prices_indexed = prices.set_index('open_time')
    
    # Resample to 4h
    df_4h = prices_indexed.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    if len(df_4h) < 100:
        return np.zeros(n)
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # 4h Supertrend for trend direction
    _, supertrend_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h indicators back to 1h timeframe using reindex with ffill
    supertrend_dir_4h_series = pd.Series(supertrend_dir_4h, index=df_4h.index)
    supertrend_dir_4h_mapped = supertrend_dir_4h_series.reindex(prices_indexed.index, method='ffill').fillna(0).values
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Minimum bars for valid signals
    first_valid = max(200, 48 * 4, 26, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend_dir_4h_mapped[i]) or np.isnan(macd_hist_1h[i]):
            signals[i] = 0.0
            continue
        
        supertrend_trend = supertrend_dir_4h_mapped[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        macd_hist = macd_hist_1h[i]
        dema8 = dema8_1h[i]
        dema21 = dema21_1h[i]
        
        # Z-score filter - avoid extreme entries
        if abs(zscore_val) > ZSCORE_MAX:
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
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
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
        
        # Entry logic: 4h Supertrend trend + 1h DEMA crossover + MACD confirmation
        if supertrend_trend == 1:  # Bullish trend on 4h
            # DEMA crossover (fast above slow) + MACD histogram positive
            if dema8 > dema21 and macd_hist > 0:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                    
        elif supertrend_trend == -1:  # Bearish trend on 4h
            # DEMA crossover (fast below slow) + MACD histogram negative
            if dema8 < dema21 and macd_hist < 0:
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