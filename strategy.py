#!/usr/bin/env python3
"""
EXPERIMENT #034 - MTF DEMA+Donchian+MACD+RSI+ATR Dynamic Sizing (4h+1h v1)
==================================================================================================
Hypothesis: Current best uses 15m+1h+4h. Let's try 4h trend + 1h entries with:
- DEMA (faster than HMA, less lag) for trend direction
- Donchian channels for breakout confirmation (20-period high/low)
- MACD histogram for momentum confirmation
- ATR-based DYNAMIC position sizing: size = base * (target_ATR_pct / current_ATR_pct)
- Tighter stoploss: 1.5*ATR (better R:R ratio than 2.0*ATR)
- RSI thresholds: 35-65 (wider than 40-60 for more entries)

Why this should beat current best (Sharpe=3.653):
- 4h trend is more stable than 1h (fewer whipsaws)
- 1h entries give good timing without 15m noise
- Donchian adds breakout confirmation (not tried in recent failures)
- Dynamic ATR sizing reduces position when volatility is high (better DD control)
- DEMA is more responsive than HMA for trend changes

Position sizing: base=0.30, adjusted by ATR ratio (target 2% ATR / current ATR%)
Stoploss: 1.5*ATR (tighter for better risk management)
Take profit: 2R with trail at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_dema_donchian_macd_rsi_atr_dynamic_4h_1h_v1"
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
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = pd.Series(close).ewm(span=period, min_periods=period).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period).mean().values
    
    dema = 2 * ema1 - ema2
    dema[:period] = 0
    
    return dema


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands)"""
    n = len(close) if 'close' in dir() else len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
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
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    macd_hist = macd_line - macd_signal
    
    macd_line[:slow] = 0
    macd_signal[:slow + signal] = 0
    macd_hist[:slow + signal] = 0
    
    return macd_line, macd_signal, macd_hist


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


def calculate_atr_pct(atr, close):
    """Calculate ATR as percentage of price"""
    n = len(close)
    atr_pct = np.zeros(n)
    for i in range(n):
        if close[i] > 0:
            atr_pct[i] = atr[i] / close[i]
    return atr_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    atr_pct_1h = calculate_atr_pct(atr_1h, close)
    rsi_1h = calculate_rsi(close, period=14)
    dema_1h = calculate_dema(close, period=21)
    donchian_upper_1h, donchian_mid_1h, donchian_lower_1h = calculate_donchian(high, low, period=20)
    macd_line_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Get 4h data using mtf_data helper (CRITICAL for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators for trend
        dema_4h_raw = calculate_dema(close_4h, period=21)
        donchian_upper_4h_raw, donchian_mid_4h_raw, donchian_lower_4h_raw = calculate_donchian(high_4h, low_4h, period=20)
        macd_hist_4h_raw = calculate_macd(close_4h, fast=12, slow=26, signal=9)[2]
        atr_4h_raw = calculate_atr(high_4h, low_4h, close_4h, period=14)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        dema_4h = align_htf_to_ltf(prices, df_4h, dema_4h_raw)
        donchian_upper_4h = align_htf_to_ltf(prices, df_4h, donchian_upper_4h_raw)
        donchian_lower_4h = align_htf_to_ltf(prices, df_4h, donchian_lower_4h_raw)
        macd_hist_4h = align_htf_to_ltf(prices, df_4h, macd_hist_4h_raw)
        atr_4h = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
        
    except Exception as e:
        # Fallback if mtf_data fails - use synthetic resampling
        bars_per_4h = 4
        n_4h = n // bars_per_4h
        
        close_4h = np.array([close[(i + 1) * bars_per_4h - 1] for i in range(n_4h)])
        high_4h = np.array([np.max(high[i * bars_per_4h:(i + 1) * bars_per_4h]) for i in range(n_4h)])
        low_4h = np.array([np.min(low[i * bars_per_4h:(i + 1) * bars_per_4h]) for i in range(n_4h)])
        
        dema_4h_raw = calculate_dema(close_4h, period=21)
        donchian_upper_4h_raw, _, donchian_lower_4h_raw = calculate_donchian(high_4h, low_4h, period=20)
        macd_hist_4h_raw = calculate_macd(close_4h, fast=12, slow=26, signal=9)[2]
        
        dema_4h = np.zeros(n)
        donchian_upper_4h = np.zeros(n)
        donchian_lower_4h = np.zeros(n)
        macd_hist_4h = np.zeros(n)
        
        for i in range(n):
            idx_4h = i // bars_per_4h
            if idx_4h < n_4h and idx_4h > 0:
                dema_4h[i] = dema_4h_raw[idx_4h - 1]
                donchian_upper_4h[i] = donchian_upper_4h_raw[idx_4h - 1]
                donchian_lower_4h[i] = donchian_lower_4h_raw[idx_4h - 1]
                macd_hist_4h[i] = macd_hist_4h_raw[idx_4h - 1]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DYNAMIC based on ATR (CRITICAL for drawdown control)
    BASE_SIZE = 0.30
    TARGET_ATR_PCT = 0.02  # Target 2% ATR
    MIN_SIZE = 0.15
    MAX_SIZE = 0.40
    
    # RSI thresholds for pullback entries (wider range for more entries)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # MACD histogram threshold for momentum
    MACD_MIN = 0
    
    # ATR stoploss multiplier (TIGHTER for better R:R)
    ATR_STOP_MULT = 1.5
    
    # Donchian breakout confirmation
    DONCHIAN_BREAKOUT_MULT = 0.5  # Price must be in top/bottom 50% of channel
    
    first_valid = max(100, 40, 26 + 9, 20, 21)
    
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
        
        # 4h trend filters
        trend_4h = 0
        if close[i] > dema_4h[i] and dema_4h[i] > 0:
            trend_4h = 1
        elif close[i] < dema_4h[i] and dema_4h[i] > 0:
            trend_4h = -1
        
        # Donchian position within channel (4h)
        donchian_range_4h = donchian_upper_4h[i] - donchian_lower_4h[i]
        if donchian_range_4h > 0:
            donchian_position = (close[i] - donchian_lower_4h[i]) / donchian_range_4h
        else:
            donchian_position = 0.5
        
        # MACD momentum (4h)
        macd_momentum_4h = 1 if macd_hist_4h[i] > MACD_MIN else (-1 if macd_hist_4h[i] < -MACD_MIN else 0)
        
        # 1h entry signals
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Dynamic position sizing based on ATR
        current_atr_pct = atr_pct_1h[i]
        if current_atr_pct > 0:
            atr_ratio = TARGET_ATR_PCT / current_atr_pct
            position_size = BASE_SIZE * atr_ratio
            position_size = np.clip(position_size, MIN_SIZE, MAX_SIZE)
        else:
            position_size = BASE_SIZE
        
        half_size = position_size / 2
        
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
            
            # Stoploss check (1.5*ATR)
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
                    signals[i] = half_size
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
                    signals[i] = -half_size
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
        
        # Entry logic: 4h trend + Donchian + MACD + 1h RSI pullback
        if trend_4h == 1:  # Bullish trend on 4h
            # Donchian breakout confirmation (price in top 50% of channel)
            if donchian_position > DONCHIAN_BREAKOUT_MULT:
                # MACD momentum confirmation (4h)
                if macd_momentum_4h >= 0:
                    # RSI pullback entry (1h)
                    if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                        signals[i] = position_size
                        position_side[i] = 1
                        entry_price[i] = price
                        tp_triggered[i] = 0
                        highest_since_entry[i] = price
                        lowest_since_entry[i] = price
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend_4h == -1:  # Bearish trend on 4h
            # Donchian breakout confirmation (price in bottom 50% of channel)
            if donchian_position < (1 - DONCHIAN_BREAKOUT_MULT):
                # MACD momentum confirmation (4h)
                if macd_momentum_4h <= 0:
                    # RSI pullback entry (1h)
                    if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                        signals[i] = -position_size
                        position_side[i] = -1
                        entry_price[i] = price
                        tp_triggered[i] = 0
                        highest_since_entry[i] = price
                        lowest_since_entry[i] = price
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals