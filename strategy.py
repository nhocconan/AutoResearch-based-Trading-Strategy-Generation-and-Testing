#!/usr/bin/env python3
"""
EXPERIMENT #017 - MTF_EMA_MACD_VOL_ATR_15m_1h_4h_v1
==================================================================================================
Hypothesis: Replace KAMA/Supertrend with EMA crossover trend + MACD momentum + Volume confirmation.
4H EMA(21/55) for trend direction + 1H MACD histogram for momentum entry + 15m Volume spike filter.

Why this should work:
- EMA(21/55) crossover is a proven trend filter (simpler than KAMA, more responsive than SMA)
- MACD histogram captures momentum shifts better than RSI/Stochastic for entries
- Volume spike (>1.5x 20-bar avg) confirms institutional participation (avoids fake breakouts)
- ATR volatility filter avoids trading during extreme volatility (reduces whipsaws)
- Different signal combination from #009 (EMA vs KAMA, MACD vs Supertrend, Volume vs BBW)

Key differences from #009:
- EMA crossover instead of KAMA for trend
- MACD histogram instead of Supertrend for entry timing
- Volume confirmation instead of Bollinger Band Width
- Same proven ATR stoploss + multi-timeframe structure

Risk management:
- Position size: 0.30 max (discrete: 0.0, ±0.20, ±0.30)
- Stoploss: 2.5*ATR (signal→0)
- Take profit: 2R (reduce to half), trail at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_ema_macd_vol_atr_15m_1h_4h_v1"
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


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    multiplier = 2.0 / (period + 1)
    
    # Initialize with SMA
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (line, signal, histogram)"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD line)
    signal_line = np.zeros(n)
    multiplier = 2.0 / (signal + 1)
    
    first_valid = slow + signal - 1
    signal_line[first_valid] = np.mean(macd_line[slow:first_valid + 1])
    
    for i in range(first_valid + 1, n):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_sma = np.zeros(n)
    vol_sma[period - 1] = np.mean(volume[:period])
    
    for i in range(period, n):
        vol_sma[i] = vol_sma[i - 1] + (volume[i] - volume[i - period]) / period
    
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    vol_sma_15m = calculate_volume_sma(volume, period=20)
    
    # Get 1h data using mtf_data helper
    try:
        df_1h = get_htf_data(prices, '1h')
        c_1h = df_1h['close'].values
        
        # 1h MACD for momentum entry timing
        macd_1h, signal_1h, hist_1h = calculate_macd(c_1h, fast=12, slow=26, signal=9)
        
        # Align 1h indicators to 15m timeframe
        hist_1h_aligned = align_htf_to_ltf(prices, df_1h, hist_1h)
        macd_1h_aligned = align_htf_to_ltf(prices, df_1h, macd_1h)
    except Exception:
        hist_1h_aligned = np.zeros(n)
        macd_1h_aligned = np.zeros(n)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        
        # 4h EMA(21/55) crossover for trend direction
        ema21_4h = calculate_ema(c_4h, period=21)
        ema55_4h = calculate_ema(c_4h, period=55)
        
        # Align 4h indicators to 15m timeframe
        ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
        ema55_4h_aligned = align_htf_to_ltf(prices, df_4h, ema55_4h)
        
        # Calculate 4h trend direction (EMA21 vs EMA55)
        trend_4h = np.zeros(n)
        for i in range(n):
            idx_4h = min(i // 16, len(c_4h) - 1)
            if idx_4h < len(c_4h) and idx_4h >= 0:
                if ema21_4h[idx_4h] > ema55_4h[idx_4h]:
                    trend_4h[i] = 1
                elif ema21_4h[idx_4h] < ema55_4h[idx_4h]:
                    trend_4h[i] = -1
    except Exception:
        ema21_4h_aligned = np.zeros(n)
        ema55_4h_aligned = np.zeros(n)
        trend_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # MACD histogram thresholds for entry timing (1h)
    MACD_LONG_MIN = -50  # Histogram rising from negative
    MACD_SHORT_MAX = 50  # Histogram falling from positive
    
    # Volume spike threshold (15m)
    VOL_SPIKE_MULT = 1.5  # Volume must be >1.5x 20-bar average
    
    # ATR volatility filter (avoid extreme volatility)
    ATR_VOLATILITY_WINDOW = 50
    ATR_VOLATILITY_UPPER = 2.0  # Don't trade if ATR > 2x recent average
    ATR_VOLATILITY_LOWER = 0.3  # Don't trade if ATR < 0.3x recent average (dead market)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 55, 20, 14)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Calculate ATR volatility filter
    atr_avg = np.zeros(n)
    for i in range(ATR_VOLATILITY_WINDOW, n):
        atr_avg[i] = np.mean(atr_15m[max(0, i - ATR_VOLATILITY_WINDOW):i])
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(vol_sma_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get aligned MTF values
        trend_4h_val = trend_4h[i] if i < len(trend_4h) else 0
        hist_1h = hist_1h_aligned[i] if i < len(hist_1h_aligned) else 0
        macd_1h = macd_1h_aligned[i] if i < len(macd_1h_aligned) else 0
        
        # ATR volatility filter
        if atr_avg[i] > 0:
            atr_ratio = atr_15m[i] / atr_avg[i]
            if atr_ratio > ATR_VOLATILITY_UPPER or atr_ratio < ATR_VOLATILITY_LOWER:
                signals[i] = 0.0
                position_side[i] = 0
                continue
        
        # Volume filter (only trade on volume spikes)
        if vol_sma_15m[i] > 0:
            vol_ratio = volume[i] / vol_sma_15m[i]
            if vol_ratio < VOL_SPIKE_MULT:
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
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
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
        
        # Entry logic: 4h trend + 1h MACD momentum + 15m Volume spike
        price = close[i]
        
        if trend_4h_val == 1:  # Bullish trend on 4h (EMA21 > EMA55)
            # MACD histogram rising from negative (momentum shift)
            # Check previous bar histogram was more negative
            if i > 0 and hist_1h < 0:
                prev_hist = hist_1h_aligned[i - 1] if i - 1 < len(hist_1h_aligned) else 0
                if prev_hist < hist_1h and hist_1h > MACD_LONG_MIN:
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                
        elif trend_4h_val == -1:  # Bearish trend on 4h (EMA21 < EMA55)
            # MACD histogram falling from positive (momentum shift)
            # Check previous bar histogram was more positive
            if i > 0 and hist_1h > 0:
                prev_hist = hist_1h_aligned[i - 1] if i - 1 < len(hist_1h_aligned) else 0
                if prev_hist > hist_1h and hist_1h < MACD_SHORT_MAX:
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