#!/usr/bin/env python3
"""
EXPERIMENT #007 - MTF DEMA+MACD+BBW+Daily Filter (1h+4h+1d v1)
==================================================================================================
Hypothesis: Switch to 1h base timeframe (reduces noise vs 15m, lower transaction costs) + 
4h DEMA crossover (faster trend detection than KAMA) + 1d SMA-50 filter (higher TF confirmation) + 
MACD histogram entry timing (proven momentum signal) + Bollinger Band Width regime filter 
(only trade when volatility is expanding from squeeze).

Why this should beat #006:
- 1h base has fewer false signals than 15m (current best uses 15m but higher TF may reduce churn)
- DEMA reacts faster than KAMA/EMA to trend changes (double smoothing removes lag)
- Daily SMA-50 filter prevents counter-trend trades in strong macro trends
- MACD histogram crosses are cleaner entry signals than RSI pullbacks
- BBW regime filter avoids trading during extreme compression/expansion

Risk Management:
- Signal size: 0.0, ±0.20, ±0.30, ±0.35 (discrete levels)
- Stoploss: 2*ATR trailing stop (signal→0 when breached)
- Take profit: 2R (reduce to half position)
- Leverage: 1.0x (no leverage, position sizing controls risk)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_dema_macd_bbw_daily_filter_1h_4h_1d_v1"
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


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average (DEMA)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    dema = 2 * ema1 - ema2
    dema[:period] = np.nan
    
    return dema


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD (line, signal, histogram)"""
    n = len(close)
    if n < slow + signal_period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal_period, adjust=False, min_periods=signal_period).mean().values
    macd_hist = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_hist


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands (upper, middle, lower, bandwidth)"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    # Bandwidth = (upper - lower) / middle
    bandwidth = np.zeros(n)
    for i in range(period - 1, n):
        if middle[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bandwidth[i] = 0
    
    return upper, middle, lower, bandwidth


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_percentile_rank(values, period=100):
    """Calculate percentile rank of current value vs last N periods"""
    n = len(values)
    percentile = np.zeros(n)
    
    for i in range(period - 1, n):
        window = values[i - period + 1:i + 1]
        current = values[i]
        rank = np.sum(window <= current) / period
        percentile[i] = rank
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    macd_line_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal_period=9)
    bb_upper, bb_middle, bb_lower, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_percentile_1h = calculate_percentile_rank(bbw_1h, period=100)
    rsi_1h = calculate_rsi(close, period=14)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h DEMA crossover for trend direction
        dema_fast_4h = calculate_dema(c_4h, period=8)
        dema_slow_4h = calculate_dema(c_4h, period=21)
        
        # Align 4h indicators to 1h timeframe
        dema_fast_4h_aligned = align_htf_to_ltf(prices, df_4h, dema_fast_4h)
        dema_slow_4h_aligned = align_htf_to_ltf(prices, df_4h, dema_slow_4h)
        c_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
        
    except Exception:
        dema_fast_4h_aligned = np.zeros(n)
        dema_slow_4h_aligned = np.zeros(n)
        c_4h_aligned = np.zeros(n)
    
    # Get 1d data using mtf_data helper for macro filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # 1d SMA-50 for macro trend filter
        sma_50_1d = pd.Series(c_1d).rolling(window=50, min_periods=50).mean().values
        
        # Align 1d indicators to 1h timeframe
        sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
        c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
        
    except Exception:
        sma_50_1d_aligned = np.zeros(n)
        c_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.20
    
    # MACD histogram thresholds
    MACD_HIST_MIN = 0  # Must be positive for long, negative for short
    
    # BBW regime filter (only trade when volatility is expanding from squeeze)
    BBW_PERCENTILE_MIN = 0.30  # Not in extreme squeeze
    BBW_PERCENTILE_MAX = 0.85  # Not in extreme expansion
    
    # RSI filter (avoid extreme levels)
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 50, 100, 26 + 9)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for invalid data
        if np.isnan(atr_1h[i]) or np.isnan(macd_hist_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        price = close[i]
        
        # Get aligned MTF values
        dema_fast_4h_val = dema_fast_4h_aligned[i] if i < len(dema_fast_4h_aligned) else 0
        dema_slow_4h_val = dema_slow_4h_aligned[i] if i < len(dema_slow_4h_aligned) else 0
        c_4h_val = c_4h_aligned[i] if i < len(c_4h_aligned) else 0
        sma_50_1d_val = sma_50_1d_aligned[i] if i < len(sma_50_1d_aligned) else 0
        c_1d_val = c_1d_aligned[i] if i < len(c_1d_aligned) else 0
        
        # 4h DEMA trend direction
        trend_4h = 0
        if dema_fast_4h_val > 0 and dema_slow_4h_val > 0:
            if dema_fast_4h_val > dema_slow_4h_val and c_4h_val > dema_fast_4h_val:
                trend_4h = 1
            elif dema_fast_4h_val < dema_slow_4h_val and c_4h_val < dema_fast_4h_val:
                trend_4h = -1
        
        # 1d SMA-50 macro filter
        macro_trend = 0
        if sma_50_1d_val > 0 and c_1d_val > 0:
            if c_1d_val > sma_50_1d_val:
                macro_trend = 1
            elif c_1d_val < sma_50_1d_val:
                macro_trend = -1
        
        # BBW regime filter
        bbw_pct = bbw_percentile_1h[i] if i < len(bbw_percentile_1h) else 0
        bbw_ok = BBW_PERCENTILE_MIN <= bbw_pct <= BBW_PERCENTILE_MAX
        
        # MACD histogram value
        macd_hist_val = macd_hist_1h[i] if i < len(macd_hist_1h) else 0
        
        # RSI value
        rsi_val = rsi_1h[i] if i < len(rsi_1h) else 0
        
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
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
                    if price > trail_stop:
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
        
        # Entry logic: 4h DEMA trend + 1d macro filter + 1h MACD entry + BBW regime + RSI filter
        # All filters must agree for entry
        
        # MACD histogram cross detection (current > 0, previous <= 0 for long)
        macd_hist_prev = macd_hist_1h[i - 1] if i > 0 else 0
        
        if trend_4h == 1 and macro_trend >= 0:  # Bullish 4h trend, neutral/bullish daily
            # MACD histogram turning positive (cross above 0)
            # BBW in normal regime (not squeeze, not extreme expansion)
            # RSI not overbought
            if (macd_hist_val > MACD_HIST_MIN and macd_hist_prev <= MACD_HIST_MIN and
                bbw_ok and
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1 and macro_trend <= 0:  # Bearish 4h trend, neutral/bearish daily
            # MACD histogram turning negative (cross below 0)
            # BBW in normal regime
            # RSI not oversold
            if (macd_hist_val < -MACD_HIST_MIN and macd_hist_prev >= -MACD_HIST_MIN and
                bbw_ok and
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
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