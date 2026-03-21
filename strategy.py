#!/usr/bin/env python3
"""
EXPERIMENT #020 - MTF HMA Trend + RSI Pullback + Z-Score (30m+4h+1d v1)
==================================================================================================
Hypothesis: Simplify the multi-timeframe approach using 30m base (more trades than 15m, less noise than 5m).
Key differences from #004:
- 30m primary timeframe (proven in #031, #034, #035 success)
- 4h HMA trend filter (simpler, more reliable than Supertrend)
- 1d SMA(50) for major trend direction (from successful #009)
- Z-score(20) regime filter instead of BBW (better at detecting extremes)
- Simpler position management (no complex state tracking bugs)

Why this should work:
- 30m has shown success in multiple experiments
- Daily SMA(50) from #009 (Sharpe=0.065) proven effective
- Z-score filter avoids choppy markets better than BBW
- Cleaner signal logic reduces bugs and whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_zscore_30m_4h_1d_v1"
timeframe = "30m"
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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
        else:
            zscore[i] = 0
    
    return zscore


def calculate_sma(close, period=50):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    
    return sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 30m indicators for entry timing
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    zscore_30m = calculate_zscore(close, period=20)
    hma_30m = calculate_hma(close, period=21)
    
    # Get 4h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(c_4h, period=21)
        
        # Align 4h indicators to 30m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_aligned = np.zeros(n)
        close_4h_aligned = np.zeros(n)
    
    # Get 1d data using mtf_data helper for major trend filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # Daily SMA(50) for major trend (from successful #009)
        sma_1d = calculate_sma(c_1d, period=50)
        
        # Align daily indicators to 30m timeframe
        sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    except Exception:
        # Fallback if mtf_data fails
        sma_1d_aligned = np.zeros(n)
        close_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.0875
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score thresholds for regime filter (avoid extremes)
    ZSCORE_MAX = 1.5
    ZSCORE_MIN = -1.5
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 50, 14 * 2, 20)
    
    # Track position state for stoploss/takeprofit
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_level_reached = np.zeros(n)
    
    for i in range(first_valid, n):
        # Validate data
        if np.isnan(atr_30m[i]) or np.isnan(rsi_30m[i]) or np.isnan(zscore_30m[i]):
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        if atr_30m[i] == 0 or hma_4h_aligned[i] == 0 or sma_1d_aligned[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        price = close[i]
        current_atr = atr_30m[i]
        
        # Get MTF trend signals
        trend_4h = 0
        if close_4h_aligned[i] > hma_4h_aligned[i]:
            trend_4h = 1
        elif close_4h_aligned[i] < hma_4h_aligned[i]:
            trend_4h = -1
        
        trend_1d = 0
        if close_1d_aligned[i] > sma_1d_aligned[i]:
            trend_1d = 1
        elif close_1d_aligned[i] < sma_1d_aligned[i]:
            trend_1d = -1
        
        # Z-score regime filter (avoid extreme overbought/oversold)
        zscore_valid = (ZSCORE_MIN <= zscore_30m[i] <= ZSCORE_MAX)
        
        # Check existing position for stoploss/takeprofit
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1]
            prev_tp = tp_level_reached[i - 1]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(highest_since_entry[i - 1], price) if highest_since_entry[i - 1] > 0 else price
                current_low = min(lowest_since_entry[i - 1], price) if lowest_since_entry[i - 1] > 0 else price
            else:
                current_high = max(highest_since_entry[i - 1], price) if highest_since_entry[i - 1] > 0 else price
                current_low = min(lowest_since_entry[i - 1], price) if lowest_since_entry[i - 1] > 0 else price
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR from entry)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * current_atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    tp_level_reached[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * current_atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    tp_level_reached[i] = 1
                    continue
                
                # Trail stop at 1R after TP reached
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * current_atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        tp_level_reached[i] = 0
                        continue
            
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * current_atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    tp_level_reached[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * current_atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    tp_level_reached[i] = 1
                    continue
                
                # Trail stop at 1R after TP reached
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * current_atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        tp_level_reached[i] = 0
                        continue
            
            # Check if trend reversed - close position
            if prev_side == 1 and trend_4h != 1:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                tp_level_reached[i] = 0
                continue
            
            if prev_side == -1 and trend_4h != -1:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                tp_level_reached[i] = 0
                continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            tp_level_reached[i] = tp_level_reached[i - 1]
            continue
        
        # No existing position - check for new entry
        # Entry logic: 4h trend + 1d trend agreement + 30m RSI pullback + Z-score filter
        
        if trend_4h == 1 and trend_1d == 1 and zscore_valid:  # Bullish alignment
            if RSI_LONG_MIN <= rsi_30m[i] <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                tp_level_reached[i] = 0
                
        elif trend_4h == -1 and trend_1d == -1 and zscore_valid:  # Bearish alignment
            if RSI_SHORT_MIN <= rsi_30m[i] <= RSI_SHORT_MAX:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                tp_level_reached[i] = 0
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals