#!/usr/bin/env python3
"""
EXPERIMENT #014 - MTF KAMA+Donchian+RSI (4h+1d v1)
==================================================================================================
Hypothesis: Use 4h primary timeframe with 1d trend filter. KAMA adapts to volatility better than
fixed MAs, Donchian Channel confirms breakout momentum, and daily trend ensures we trade with
higher timeframe direction. This differs from current best by:
- KAMA (adaptive) instead of Supertrend/HMA for trend
- Donchian Channel breakout confirmation (20-period high/low)
- 4h primary (cleaner signals than 15m/30m, more trades than 6h/12h)
- 1d trend filter (stronger than 4h trend alone)

Why this should work:
- KAMA reduces whipsaws in choppy markets (ER-based adaptation)
- Donchian breakout confirms momentum (avoids false entries)
- 4h timeframe balances trade frequency with signal quality
- Daily trend filter prevents counter-trend trades (proven in #009, #010)
- Fewer signal changes = lower transaction costs
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_donchian_rsi_4h_1d_v1"
timeframe = "4h"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    change = np.zeros(n)
    volatility = np.zeros(n)
    er = np.zeros(n)
    sc = np.zeros(n)
    
    for i in range(period, n):
        change[i] = abs(close[i] - close[i - period])
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i - period):i + 1])))
        
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
        
        sc[i] = (er[i] * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower


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
    
    # 4h indicators for entry timing
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    kama_4h = calculate_kama(close, period=10, fast=2, slow=30)
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(high, low, period=20)
    
    # Get 1d data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        
        # 1d indicators for trend filter
        kama_1d = calculate_kama(c_1d, period=10, fast=2, slow=30)
        sma_1d = calculate_sma(c_1d, period=50)
        rsi_1d = calculate_rsi(c_1d, period=14)
        
        # Align 1d indicators to 4h timeframe (auto shift for completed bars)
        kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
        sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    except Exception:
        # Fallback if mtf_data fails
        kama_1d_aligned = np.zeros(n)
        sma_1d_aligned = np.zeros(n)
        rsi_1d_aligned = np.zeros(n) + 50  # Neutral RSI
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Donchian breakout confirmation
    DONCHIAN_BREAKOUT_CONFIRM = 0.005  # 0.5% above/below channel
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 50, 30, 20, 14)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_4h[i]) or np.isnan(rsi_4h[i]) or atr_4h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values
        kama_1d_val = kama_1d_aligned[i] if i < len(kama_1d_aligned) else 0
        sma_1d_val = sma_1d_aligned[i] if i < len(sma_1d_aligned) else 0
        rsi_1d_val = rsi_1d_aligned[i] if i < len(rsi_1d_aligned) else 50
        
        # 1d trend filter (KAMA vs SMA50 + RSI direction)
        trend_1d = 0
        if kama_1d_val > 0 and sma_1d_val > 0:
            if kama_1d_val > sma_1d_val and rsi_1d_val > 50:
                trend_1d = 1  # Bullish
            elif kama_1d_val < sma_1d_val and rsi_1d_val < 50:
                trend_1d = -1  # Bearish
        
        # Skip if no clear daily trend
        if trend_1d == 0:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                # Close position if trend changes
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
            else:
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
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_4h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_4h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_4h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_4h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_4h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_4h[i]
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
        
        # Entry logic: 1d trend + 4h KAMA + Donchian breakout + RSI pullback
        price = close[i]
        
        # 4h KAMA trend direction
        kama_trend_4h = 0
        if i > 10 and kama_4h[i] > 0 and kama_4h[i - 1] > 0:
            if kama_4h[i] > kama_4h[i - 1]:
                kama_trend_4h = 1
            elif kama_4h[i] < kama_4h[i - 1]:
                kama_trend_4h = -1
        
        # Donchian breakout confirmation
        donchian_breakout = 0
        if donchian_upper_4h[i] > 0 and donchian_lower_4h[i] > 0:
            if price > donchian_upper_4h[i] * (1 + DONCHIAN_BREAKOUT_CONFIRM):
                donchian_breakout = 1
            elif price < donchian_lower_4h[i] * (1 - DONCHIAN_BREAKOUT_CONFIRM):
                donchian_breakout = -1
        
        if trend_1d == 1 and kama_trend_4h == 1 and donchian_breakout == 1:
            # Bullish: Daily trend + 4h KAMA up + Donchian breakout
            # RSI pullback (not overbought)
            if RSI_LONG_MIN <= rsi_4h[i] <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_1d == -1 and kama_trend_4h == -1 and donchian_breakout == -1:
            # Bearish: Daily trend + 4h KAMA down + Donchian breakout
            # RSI pullback (not oversold)
            if RSI_SHORT_MIN <= rsi_4h[i] <= RSI_SHORT_MAX:
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