#!/usr/bin/env python3
"""
EXPERIMENT #016 - KAMA Trend + MACD Entry + Bollinger Regime Filter
====================================================================================
Hypothesis: KAMA adapts to volatility better than HMA/EMA, reducing whipsaws in choppy markets.
MACD histogram crosses provide clearer momentum entry signals than RSI pullbacks.
Bollinger Band Width filter avoids trading during extreme squeezes (low vol breakouts often fail).

Key differences from #021:
- KAMA(10,2,30) instead of HMA - adapts to market efficiency ratio
- MACD histogram cross instead of RSI - momentum-based entries
- Bollinger BW percentile filter instead of Z-score - regime detection
- 1h timeframe instead of 15m - fewer trades, less fee churn
- 4h KAMA trend + 1h MACD entries (proven MTF combo from #008)

Why this might beat Sharpe=5.525:
- KAMA reduces noise in ranging markets vs fixed-period MAs
- MACD histogram captures momentum shifts earlier than RSI
- BBW filter avoids low-volatility breakout traps
- 1h timeframe balances signal quality vs trade frequency
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_bbw_v2"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.zeros(n)
    
    if n < slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # KAMA calculation
    kama[er_period - 1] = close[er_period - 1]
    for i in range(er_period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    if n < slow + signal_period:
        return macd_line, signal_line, histogram
    
    # EMA calculation helper
    def ema(arr, period):
        result = np.zeros(len(arr))
        multiplier = 2 / (period + 1)
        result[period - 1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (arr[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line is EMA of MACD
    valid_start = slow + signal_period - 1
    for i in range(valid_start, n):
        if i == valid_start:
            signal_line[i] = np.mean(macd_line[slow - 1:i + 1])
        else:
            multiplier = 2 / (signal_period + 1)
            signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)
    
    if n < period:
        return middle, upper, lower, bandwidth
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        if middle[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bandwidth[i] = 0
    
    return middle, upper, lower, bandwidth


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    macd_1h, signal_1h, hist_1h = calculate_macd(close, fast=12, slow=26, signal_period=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_mid_1h, bb_upper_1h, bb_lower_1h, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    kama_fast_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow_1h = calculate_kama(close, er_period=10, fast_period=5, slow_period=50)
    
    # 4h KAMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    
    # Calculate 4h KAMA for trend
    kama_fast_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    kama_slow_4h = calculate_kama(c_4h, er_period=10, fast_period=5, slow_period=50)
    
    # 4h trend direction based on KAMA cross and price position
    trend_4h = np.zeros(len(c_4h))
    for i in range(50, len(c_4h)):
        if kama_fast_4h[i] > kama_slow_4h[i] and c_4h[i] > kama_fast_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif kama_fast_4h[i] < kama_slow_4h[i] and c_4h[i] < kama_fast_4h[i]:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Calculate BBW percentile for regime filter (rolling 100-period)
    bbw_percentile = np.zeros(n)
    bbw_window = 100
    for i in range(bbw_window - 1, n):
        window = bbw_1h[i - bbw_window + 1:i + 1]
        bbw_percentile[i] = np.sum(window <= bbw_1h[i]) / bbw_window * 100
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position (MAX 0.40 per rules)
    SIZE_HALF = 0.20   # Half position (after take profit)
    
    # MACD histogram thresholds for momentum entries
    MACD_HIST_THRESHOLD = 0  # Cross above/below zero
    MACD_HIST_MIN = 50  # Minimum histogram value for strong momentum
    
    # Bollinger Band Width regime filter
    BBW_MIN_PERCENTILE = 20  # Avoid extreme squeezes (<20th percentile)
    BBW_MAX_PERCENTILE = 85  # Avoid extreme expansions (>85th percentile)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    first_valid = max(50, 35, 20, bbw_window)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    
    for i in range(first_valid, n):
        if np.isnan(macd_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bbw_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        macd_hist = hist_1h[i]
        macd_hist_prev = hist_1h[i - 1] if i > 0 else 0
        bbw_pct = bbw_percentile[i]
        atr = atr_1h[i]
        price = close[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            continue
        
        # BBW regime filter - avoid extreme squeezes and expansions
        if bbw_pct < BBW_MIN_PERCENTILE or bbw_pct > BBW_MAX_PERCENTILE:
            # If in position, hold; otherwise stay flat
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            
            if prev_side == 1:  # Long position
                # Stoploss check
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    # Reduce to half position
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # MACD histogram reversal exit
                if macd_hist < MACD_HIST_THRESHOLD and macd_hist_prev >= MACD_HIST_THRESHOLD:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
            elif prev_side == -1:  # Short position
                # Stoploss check
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    # Reduce to half position
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # MACD histogram reversal exit
                if macd_hist > MACD_HIST_THRESHOLD and macd_hist_prev <= MACD_HIST_THRESHOLD:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
            
            # Hold position if no exit signal
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            continue
        
        # MACD histogram cross entry signals
        if trend == 1:  # 4h uptrend
            # Long entry: MACD histogram crosses above zero with momentum
            if macd_hist > MACD_HIST_THRESHOLD and macd_hist_prev <= MACD_HIST_THRESHOLD:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # Short entry: MACD histogram crosses below zero with momentum
            if macd_hist < MACD_HIST_THRESHOLD and macd_hist_prev >= MACD_HIST_THRESHOLD:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
    
    return signals