#!/usr/bin/env python3
"""
EXPERIMENT #044 - Bollinger Mean Reversion with Daily Trend Filter (4h Primary)
==================================================================================================
Hypothesis: Current best (Sharpe=0.537) uses 4h HMA+RSI pullback on 1h primary. This tries 4h PRIMARY
with Bollinger Band mean reversion entries + 1d trend filter + RSI confirmation.

Key innovations:
1. 4h PRIMARY + 1d HTF: Cleaner signals than 1h/30m, more trades than daily
2. Bollinger mean reversion: Buy at lower band in uptrend, sell at upper band in downtrend
3. RSI(14) confirmation: Only enter when RSI confirms oversold/overbought conditions
4. Fixed position sizing: 0.30 for full position, 0.15 for half (discrete levels, low churn)
5. 2.0*ATR stoploss: More room than 1.5*ATR to avoid premature stops

Why this should beat kama_macd_momentum_mtf_1h_4h_1d_v1 (Sharpe=0.290):
- 4h timeframe has less noise than 1h for mean reversion strategies
- Bollinger bands capture volatility-based entry points better than KAMA/MACD
- Single HTF (1d) reduces complexity and overfitting vs 4h+1d filters
- Fixed sizing avoids volatility-adjustment instability
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "bb_rsi_meanreversion_daily_trend_4h_v1"
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """
    Bollinger Bands - mean reversion indicator
    Returns: upper_band, middle_band, lower_band, bandwidth
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    
    middle = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / middle
    
    # Handle NaN/inf
    upper = np.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0)
    lower = np.nan_to_num(lower, nan=0.0, posinf=0.0, neginf=0.0)
    bandwidth = np.nan_to_num(bandwidth, nan=0.0, posinf=0.0, neginf=0.0)
    
    return upper, middle, lower, bandwidth


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    rsi = np.zeros(n)
    
    # Calculate price changes
    delta = np.diff(close)
    
    # Separate gains and losses
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    for i in range(1, n):
        if delta[i - 1] > 0:
            gains[i] = delta[i - 1]
        else:
            losses[i] = -delta[i - 1]
    
    # Wilder's smoothing for first RSI
    avg_gain = np.mean(gains[1:period + 1])
    avg_loss = np.mean(losses[1:period + 1])
    
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    
    # Continue with Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_hma(close, period=21):
    """
    Hull Moving Average - smoother and more responsive than EMA
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close_series, half_period)
    wma_full = wma(close_series, period)
    
    wma_diff = 2 * wma_half - wma_full
    hma = wma(wma_diff, sqrt_period)
    
    hma_values = hma.values
    hma_values = np.nan_to_num(hma_values, nan=0.0, posinf=0.0, neginf=0.0)
    
    return hma_values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 4h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_4h = calculate_atr(high, low, close, period=14)
    upper_4h, middle_4h, lower_4h, bw_4h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    rsi_4h = calculate_rsi(close, period=14)
    hma_4h = calculate_hma(close, period=21)
    
    # ========== 1d INDICATORS (LONG-TERM TREND) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        hma_1d = calculate_hma(close_1d, period=21)
        atr_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
        rsi_1d = calculate_rsi(close_1d, period=14)
        
        # Align to 4h timeframe (auto shift for completed bars)
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
        
    except Exception:
        hma_1d_aligned = np.zeros(n)
        rsi_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - FIXED discrete levels
    SIZE_FULL = 0.30      # Full position (30% of capital)
    SIZE_HALF = 0.15      # Half position (15% of capital)
    
    # Stoploss and take profit
    ATR_STOP_MULT = 2.0   # 2.0*ATR stoploss
    TP_MULT = 2.0         # 2R take profit
    TRAIL_MULT = 1.0      # Trail at 1R
    
    # RSI thresholds for mean reversion
    RSI_OVERSOLD = 35     # Buy when RSI < 35 in uptrend
    RSI_OVERBOUGHT = 65   # Sell when RSI > 65 in downtrend
    
    first_valid = 100
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_4h[i]) or atr_4h[i] == 0 or np.isnan(middle_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        
        # Bollinger Bands
        upper = upper_4h[i]
        middle = middle_4h[i]
        lower = lower_4h[i]
        
        # RSI
        rsi = rsi_4h[i]
        
        # HMA trend
        hma = hma_4h[i]
        
        # 1d trend filter
        hma_1d_val = hma_1d_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        
        # Determine 1d trend direction
        trend_1d = 0
        if hma_1d_val > 0 and price > hma_1d_val:
            trend_1d = 1
        elif hma_1d_val > 0 and price < hma_1d_val:
            trend_1d = -1
        
        # 4h trend direction
        trend_4h = 0
        if hma > 0 and price > hma:
            trend_4h = 1
        elif hma > 0 and price < hma:
            trend_4h = -1
        
        # ========== CHECK EXISTING POSITIONS ==========
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
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - TRAIL_MULT * ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + TRAIL_MULT * ATR_STOP_MULT * atr
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
        
        # ========== ENTRY LOGIC - BOLLINGER MEAN REVERSION IN TREND ==========
        # LONG: 1d trend up + price at/near lower BB + RSI oversold + 4h trend not strongly down
        long_condition = (
            trend_1d == 1 and                          # Daily trend up
            price <= lower * 1.005 and                 # Price at or below lower band (with small buffer)
            rsi < RSI_OVERSOLD and                     # RSI oversold
            trend_4h >= 0 and                          # 4h trend neutral or up
            rsi_1d_val > 40 and                        # Daily RSI not oversold (trend intact)
            middle > 0 and lower > 0                   # Valid bands
        )
        
        # SHORT: 1d trend down + price at/near upper BB + RSI overbought + 4h trend not strongly up
        short_condition = (
            trend_1d == -1 and                         # Daily trend down
            price >= upper * 0.995 and                 # Price at or above upper band (with small buffer)
            rsi > RSI_OVERBOUGHT and                   # RSI overbought
            trend_4h <= 0 and                          # 4h trend neutral or down
            rsi_1d_val < 60 and                        # Daily RSI not overbought (trend intact)
            middle > 0 and upper > 0                   # Valid bands
        )
        
        if long_condition:
            signals[i] = SIZE_FULL
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            signals[i] = -SIZE_FULL
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals