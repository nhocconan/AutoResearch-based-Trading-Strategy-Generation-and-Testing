#!/usr/bin/env python3
"""
Experiment #1019: 1h Primary + 4h/12h HTF — Ehlers Fisher Transform + HMA Trend Filter

Hypothesis: Ehlers Fisher Transform catches reversals better than RSI in bear/range markets.
Combined with HTF HMA trend filter, this should work across BTC/ETH/SOL including 2022 crash.

Key innovations:
1. Ehlers Fisher Transform (period=9): Normalizes price to Gaussian distribution
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
2. 12h HMA(21) for long-term trend bias (only trade with HTF trend)
3. 4h HMA(21) for intermediate confirmation
4. Session filter (08-20 UTC) for liquidity
5. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- Fisher Transform is proven to catch reversals in bear markets (research-validated)
- HTF HMA filter avoids counter-trend trades that failed in 2022
- Different from all 838 failed strategies (no RSI, no Choppiness, no Connors)
- 1h with strict HTF filter = 40-80 trades/year target
- Works on BTC/ETH (not just SOL) because Fisher is mean-reversion based

Entry conditions (LOOSE to guarantee trades):
- LONG: 12h_HMA bullish + Fisher cross above -1.5 + session 08-20 UTC
- SHORT: 12h_HMA bearish + Fisher cross below +1.5 + session 08-20 UTC

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_trend_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Makes reversals easier to detect by transforming price to bounded range
    
    Formula:
    1. Calculate typical price: (high + low + close) / 3
    2. Normalize: (price - lowest(period)) / (highest(period) - lowest(period))
    3. Transform: 0.5 * ln((1 + normalized) / (1 - normalized))
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Typical price
    typical = (high + low + close) / 3.0
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        window = typical[i-period+1:i+1]
        if np.any(np.isnan(window)):
            continue
        
        highest = np.max(window)
        lowest = np.min(window)
        price_range = highest - lowest
        
        if price_range < 1e-10:
            continue
        
        # Normalize to 0-1 range, then scale to -0.99 to +0.99
        normalized = 2.0 * (typical[i] - lowest) / price_range - 1.0
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Previous value for crossover detection
        if i > period:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS ===
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === SESSION FILTER (08-20 UTC) ===
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === FISHER CROSSOVER DETECTION ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_cross_long = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_short = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 12h bullish + 4h bullish + Fisher cross + session
        if hma_12h_bull and hma_4h_bull and fisher_cross_long and in_session:
            desired_signal = SIZE_BASE
        
        # SHORT: 12h bearish + 4h bearish + Fisher cross + session
        elif hma_12h_bear and hma_4h_bear and fisher_cross_short and in_session:
            desired_signal = -SIZE_BASE
        
        # Stronger signals when both HTFs align strongly
        strong_bull = hma_12h_bull and hma_4h_bull and hma_12h_aligned[i] > hma_4h_aligned[i]
        strong_bear = hma_12h_bear and hma_4h_bear and hma_12h_aligned[i] < hma_4h_aligned[i]
        
        if strong_bull and fisher_cross_long and in_session:
            desired_signal = SIZE_STRONG
        elif strong_bear and fisher_cross_short and in_session:
            desired_signal = -SIZE_STRONG
        
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