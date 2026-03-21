#!/usr/bin/env python3
"""
EXPERIMENT #021 - Simplified 1h HMA+RSI with 4h Trend Filter
==================================================================================================
Hypothesis: The current 15m+1h+4h strategy is over-engineered. A cleaner 1h primary with 4h 
trend filter should reduce whipsaws and fees while maintaining signal quality. 

Key changes from #004:
- Primary TF: 1h instead of 15m (cleaner signals, less noise)
- Simpler entry: HMA crossover + RSI pullback (remove MACD/BBW complexity)
- Only 2 timeframes: 1h entries + 4h trend (not 3)
- Discrete signal levels: 0.0, ±0.25, ±0.35 (reduce churn)
- ATR-based dynamic position sizing

Why this should beat Sharpe=0.065:
- 1h has proven success in literature for crypto swing trading
- Fewer timeframe alignments = fewer look-ahead risks
- Simpler logic = more robust across BTC/ETH/SOL
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_1h_4h_simplified_v1"
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


def calculate_sma(close, period=20):
    """Calculate Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values


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
    sma_200_1h = calculate_sma(close, period=200)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(c_4h, period=21)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
    except Exception:
        hma_4h_aligned = np.zeros(n)
        close_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_ZERO = 0.0
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45
    RSI_SHORT_ENTRY = 55
    RSI_EXIT = 50
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 48, 14 * 2, 21 * 4)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    entry_atr = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Validate data
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            if i > 0:
                position_side[i] = position_side[i - 1]
            continue
        
        # Get aligned MTF values
        hma_4h_val = hma_4h_aligned[i] if i < len(hma_4h_aligned) else 0
        close_4h_val = close_4h_aligned[i] if i < len(close_4h_aligned) else close[i]
        
        # 4h trend filter: price vs HMA
        trend_4h = 0
        if hma_4h_val > 0:
            if close_4h_val > hma_4h_val:
                trend_4h = 1
            elif close_4h_val < hma_4h_val:
                trend_4h = -1
        
        # 1h trend: HMA crossover
        hma_trend_1h = 0
        if hma_fast_1h[i] > hma_slow_1h[i]:
            hma_trend_1h = 1
        elif hma_fast_1h[i] < hma_slow_1h[i]:
            hma_trend_1h = -1
        
        # 200 SMA filter (only trade in direction of long-term trend)
        long_term_trend = 0
        if sma_200_1h[i] > 0:
            if close[i] > sma_200_1h[i]:
                long_term_trend = 1
            elif close[i] < sma_200_1h[i]:
                long_term_trend = -1
        
        price = close[i]
        
        # Check existing positions first (stoploss / takeprofit)
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_atr = entry_atr[i - 1] if entry_atr[i - 1] > 0 else atr_1h[i]
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
            
            # Stoploss check (2.5*ATR from entry)
            stoploss_triggered = False
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * prev_atr
                if price < stoploss_price:
                    stoploss_triggered = True
            else:
                stoploss_price = prev_entry + ATR_STOP_MULT * prev_atr
                if price > stoploss_price:
                    stoploss_triggered = True
            
            if stoploss_triggered:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                entry_atr[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            # Take profit check (2R) - reduce to half
            if not prev_tp:
                if prev_side == 1:
                    tp_price = prev_entry + 2 * ATR_STOP_MULT * prev_atr
                    if price >= tp_price:
                        signals[i] = SIZE_HALF
                        position_side[i] = 1
                        entry_price[i] = prev_entry
                        entry_atr[i] = prev_atr
                        tp_triggered[i] = True
                        highest_since_entry[i] = current_high
                        lowest_since_entry[i] = current_low
                        continue
                else:
                    tp_price = prev_entry - 2 * ATR_STOP_MULT * prev_atr
                    if price <= tp_price:
                        signals[i] = -SIZE_HALF
                        position_side[i] = -1
                        entry_price[i] = prev_entry
                        entry_atr[i] = prev_atr
                        tp_triggered[i] = True
                        highest_since_entry[i] = current_high
                        lowest_since_entry[i] = current_low
                        continue
            
            # Trail stop at 1R profit after TP triggered
            if prev_tp:
                trail_triggered = False
                if prev_side == 1:
                    trail_stop = current_high - ATR_STOP_MULT * prev_atr
                    if price < trail_stop:
                        trail_triggered = True
                else:
                    trail_stop = current_low + ATR_STOP_MULT * prev_atr
                    if price > trail_stop:
                        trail_triggered = True
                
                if trail_triggered:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            
            # RSI exit signal
            rsi_exit_triggered = False
            if prev_side == 1 and rsi_1h[i] > RSI_EXIT + 15:
                rsi_exit_triggered = True
            elif prev_side == -1 and rsi_1h[i] < RSI_EXIT - 15:
                rsi_exit_triggered = True
            
            if rsi_exit_triggered:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                entry_atr[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            # Trend reversal exit
            trend_reversal = False
            if prev_side == 1 and trend_4h == -1:
                trend_reversal = True
            elif prev_side == -1 and trend_4h == 1:
                trend_reversal = True
            
            if trend_reversal:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                entry_atr[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            # Hold position
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            entry_atr[i] = entry_atr[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: All filters must align
        # 4h trend + 1h HMA trend + 200 SMA direction + RSI pullback
        
        if trend_4h == 1 and hma_trend_1h == 1 and long_term_trend >= 0:
            # Bullish: RSI pullback entry
            if rsi_1h[i] <= RSI_LONG_ENTRY:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                entry_atr[i] = atr_1h[i]
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend_4h == -1 and hma_trend_1h == -1 and long_term_trend <= 0:
            # Bearish: RSI pullback entry
            if rsi_1h[i] >= RSI_SHORT_ENTRY:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                entry_atr[i] = atr_1h[i]
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals