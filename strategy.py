#!/usr/bin/env python3
"""
EXPERIMENT #010 - MTF HMA+Donchian+MACD+Zscore (15m+1h+4h v1)
==================================================================================================
Hypothesis: Replace KAMA with HMA (faster trend response), replace Supertrend with Donchian
breakout confirmation, replace RSI pullback with MACD histogram momentum entry, and use
Z-score instead of BBW for regime filter.

Key changes from #009 (Sharpe=2.274):
- 4h HMA(21/63) instead of KAMA - HMA reduces lag while maintaining smoothness
- 1h Donchian(20) breakout instead of Supertrend - cleaner breakout signals
- 15m MACD histogram cross instead of RSI pullback - momentum-based entries
- Z-score(20) filter instead of BBW - better at detecting extreme conditions
- Removed ADX filter - MACD histogram already provides momentum confirmation

Why this should beat #009:
- HMA is more responsive to trend changes than KAMA (less lag)
- Donchian breakouts capture momentum bursts better than Supertrend
- MACD histogram provides earlier entry signals than RSI pullback
- Z-score filter avoids extreme overbought/oversold conditions

Risk Management:
- Max signal: 0.28 (conservative vs 0.30 baseline)
- Discrete levels: 0.0, ±0.20, ±0.28
- Stoploss: 2.5*ATR trailing stop (slightly wider for MACD entries)
- Take profit: reduce to half at 2R, trail at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_donchian_macd_zscore_15m_1h_4h_v1"
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
    """Calculate Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA calculation helper
    def wma(data, w_period):
        result = np.zeros(len(data))
        weights = np.arange(1, w_period + 1)
        weight_sum = np.sum(weights)
        
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_input = 2 * wma_half - wma_full
    
    # Pad hma_input to match length
    hma_input_padded = np.zeros(n)
    offset = period - sqrt_period
    if offset >= 0 and offset + len(hma_input) <= n:
        hma_input_padded[offset:offset + len(hma_input)] = hma_input
    
    hma = wma(hma_input_padded, sqrt_period)
    
    return hma


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel - upper/lower bands and breakout signals"""
    n = len(close := high)  # Use high length
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    exp1 = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    exp2 = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = exp1 - exp2
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_zscore(close, period=20):
    """Calculate Z-score - measures how many standard deviations from mean"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    macd_line_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    zscore_15m = calculate_zscore(close, period=20)
    
    # Get 1h data using mtf_data helper
    df_1h = get_htf_data(prices, '1h')
    c_1h = df_1h['close'].values
    h_1h = df_1h['high'].values
    l_1h = df_1h['low'].values
    
    # 1h Donchian for breakout confirmation
    donchian_upper_1h, donchian_middle_1h, donchian_lower_1h = calculate_donchian(h_1h, l_1h, period=20)
    
    # Align 1h indicators to 15m timeframe
    donchian_upper_1h_aligned = align_htf_to_ltf(prices, df_1h, donchian_upper_1h)
    donchian_lower_1h_aligned = align_htf_to_ltf(prices, df_1h, donchian_lower_1h)
    donchian_middle_1h_aligned = align_htf_to_ltf(prices, df_1h, donchian_middle_1h)
    
    # Get 4h data using mtf_data helper for trend filter
    df_4h = get_htf_data(prices, '4h')
    c_4h = df_4h['close'].values
    
    # 4h HMA for adaptive trend
    hma_4h = calculate_hma(c_4h, period=21)
    
    # Align 4h indicators to 15m timeframe
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 4h trend direction (price vs HMA)
    trend_4h = np.zeros(n)
    for i in range(n):
        if i < len(hma_4h_aligned) and hma_4h_aligned[i] > 0:
            if c_4h[min(i // 16, len(c_4h) - 1)] > hma_4h_aligned[i]:
                trend_4h[i] = 1
            elif c_4h[min(i // 16, len(c_4h) - 1)] < hma_4h_aligned[i]:
                trend_4h[i] = -1
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.28
    SIZE_HALF = 0.14
    
    # Z-score thresholds for regime filter (avoid extremes)
    ZSCORE_MAX = 2.0
    ZSCORE_MIN = -2.0
    
    # MACD histogram threshold for momentum confirmation
    MACD_HIST_MIN = 0.0  # Must be positive for long, negative for short
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 26 + 9, 20, 21)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(macd_hist_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get aligned MTF values
        donchian_upper = donchian_upper_1h_aligned[i] if i < len(donchian_upper_1h_aligned) else 0
        donchian_lower = donchian_lower_1h_aligned[i] if i < len(donchian_lower_1h_aligned) else 0
        donchian_middle = donchian_middle_1h_aligned[i] if i < len(donchian_middle_1h_aligned) else 0
        trend_4h_val = trend_4h[i] if i < len(trend_4h) else 0
        
        # Z-score filter - avoid extreme overbought/oversold conditions
        if zscore_15m[i] > ZSCORE_MAX or zscore_15m[i] < ZSCORE_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 4h trend filter
        if trend_4h_val == 0:
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
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered - check trend still valid
            if prev_side == 1 and trend_4h_val == 1:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            elif prev_side == -1 and trend_4h_val == -1:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                # Exit if trend changes
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # Entry logic: 4h HMA trend + 1h Donchian breakout + 15m MACD histogram
        price = close[i]
        
        # Bullish entry: 4h uptrend + price above Donchian middle + MACD histogram positive
        if trend_4h_val == 1 and donchian_middle > 0:
            if price > donchian_middle and macd_hist_15m[i] > MACD_HIST_MIN:
                # Check MACD histogram increasing (momentum building)
                if i > 0 and macd_hist_15m[i] > macd_hist_15m[i - 1]:
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = False
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
        
        # Bearish entry: 4h downtrend + price below Donchian middle + MACD histogram negative
        elif trend_4h_val == -1 and donchian_middle > 0:
            if price < donchian_middle and macd_hist_15m[i] < -MACD_HIST_MIN:
                # Check MACD histogram decreasing (momentum building)
                if i > 0 and macd_hist_15m[i] < macd_hist_15m[i - 1]:
                    signals[i] = -SIZE_FULL
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = False
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals