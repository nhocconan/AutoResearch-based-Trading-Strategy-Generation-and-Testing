#!/usr/bin/env python3
"""
EXPERIMENT #008 - MTF HMA+RSI+Daily Regime (1h+4h+1d v1)
==================================================================================================
Hypothesis: Current best #004 uses 15m base with Supertrend+MACD+BBW+RSI (Sharpe=3.653).
This experiment tries a DIFFERENT combination:
- 1h as base timeframe (less noise than 15m, more responsive than 4h)
- 4h HMA(21/55) crossover for trend (smoother than Supertrend, adaptive)
- Daily SMA-50 for regime filter (only trade with daily trend - NEW)
- 1h RSI(14) pullback for entries (proven in #004)
- 1h ATR(14) for stoploss and position sizing
- Position size: 0.30 (conservative, discrete levels)
- Stoploss: 2.0*ATR (tighter than #007's 2.5*ATR)

Why this should beat #004:
- Daily regime filter adds another layer of trend confirmation
- HMA crossover is more responsive than Supertrend for trend changes
- 1h base timeframe balances noise vs signal frequency better than 15m
- Simpler indicator set (3 vs 4 in #004) may reduce overfitting
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_daily_regime_1h_4h_1d_v1"
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


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, w_period):
        result = np.zeros(len(series))
        weights = np.arange(1, w_period + 1)
        for i in range(w_period - 1, len(series)):
            window = series[i - w_period + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    raw_hma = 2 * wma_half - wma_full
    
    hma = wma(raw_hma, sqrt_period)
    
    return hma


def calculate_sma(close, period=50):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = np.zeros(n)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing (local timeframe)
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_21_1h = calculate_hma(close, period=21)
    hma_55_1h = calculate_hma(close, period=55)
    
    # Get 4h data using mtf_data helper (MANDATORY - no manual resampling)
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        
        # 4h HMA for trend confirmation
        hma_21_4h = calculate_hma(c_4h, period=21)
        hma_55_4h = calculate_hma(c_4h, period=55)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        hma_21_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_21_4h)
        hma_55_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_55_4h)
    except Exception:
        # Fallback if mtf_data fails
        hma_21_4h_aligned = np.zeros(n)
        hma_55_4h_aligned = np.zeros(n)
    
    # Get Daily data using mtf_data helper (MANDATORY)
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # Daily SMA-50 for regime filter
        sma_50_1d = calculate_sma(c_1d, period=50)
        
        # Align Daily indicators to 1h timeframe
        sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    except Exception:
        # Fallback if mtf_data fails
        sma_50_1d_aligned = np.zeros(n)
    
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
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 55, 50)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        hma_21_4h_val = hma_21_4h_aligned[i]
        hma_55_4h_val = hma_55_4h_aligned[i]
        sma_50_1d_val = sma_50_1d_aligned[i]
        
        # 1h HMA values
        hma_21_1h_val = hma_21_1h[i]
        hma_55_1h_val = hma_55_1h[i]
        
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Determine 4h HMA trend direction
        trend_4h = 0
        if hma_21_4h_val > hma_55_4h_val and hma_21_4h_val > 0:
            trend_4h = 1  # Bullish
        elif hma_21_4h_val < hma_55_4h_val and hma_55_4h_val > 0:
            trend_4h = -1  # Bearish
        
        # Determine Daily regime
        daily_regime = 0
        if sma_50_1d_val > 0:
            if price > sma_50_1d_val:
                daily_regime = 1  # Bullish regime
            elif price < sma_50_1d_val:
                daily_regime = -1  # Bearish regime
        
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
        
        # Entry logic: Daily regime + 4h HMA trend + 1h RSI pullback
        # LONG: Daily bullish + 4h HMA bullish + 1h RSI pullback
        if daily_regime == 1 and trend_4h == 1:
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        # SHORT: Daily bearish + 4h HMA bearish + 1h RSI pullback
        elif daily_regime == -1 and trend_4h == -1:
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
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