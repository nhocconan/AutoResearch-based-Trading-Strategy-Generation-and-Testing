#!/usr/bin/env python3
"""
EXPERIMENT #002 - MTF DEMA+MACD+Donchian+RSI+ADX (1h+4h v1)
==================================================================================================
Hypothesis: Current best (#040) uses 15m+1h. Try 1h+4h for cleaner trend signals with less noise.
Key changes:
- Timeframe: 1h instead of 15m (fewer but higher quality trades)
- MTF: 4h trend filter (more stable than 1h)
- DEMA(8/21) crossover for faster trend detection than HMA
- MACD histogram for momentum entry timing
- Donchian(20) breakout confirmation
- ADX(14) > 25 for trend strength filter
- RSI(14) 35-65 for pullback entries (wider range for 1h)
- Position size: 0.35 (proven safe)
- Stoploss: 2.0*ATR with trailing after 1R profit

Why this should work:
- 1h has cleaner signals than 15m (less noise)
- 4h trend is more reliable than 1h for direction
- DEMA reacts faster than HMA for entries
- MACD histogram adds momentum confirmation
- Donchian breakout ensures we're trading with momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_dema_macd_donchian_rsi_adx_1h_4h_v1"
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
    
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False).mean().values
    
    dema = 2 * ema1 - ema2
    dema[:period] = np.nan
    
    return dema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    macd_line[:slow] = np.nan
    signal_line[:slow + signal] = np.nan
    histogram[:slow + signal] = np.nan
    
    return macd_line, signal_line, histogram


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)"""
    n = len(close := high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = pd.Series(close).ewm(span=period, adjust=False).mean().values
    ema[:period] = np.nan
    
    return ema


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get 4h HTF data using mtf_data helper (MANDATORY)
    df_4h = get_htf_data(prices, '4h')
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    dema_fast_1h = calculate_dema(close, period=8)
    dema_slow_1h = calculate_dema(close, period=21)
    macd_line_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    donchian_upper_1h, donchian_lower_1h = calculate_donchian(high, low, period=20)
    adx_1h = calculate_adx(high, low, close, period=14)
    ema_50_1h = calculate_ema(close, period=50)
    
    # 4h indicators for trend (using mtf_data helper)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    dema_fast_4h = calculate_dema(close_4h, period=8)
    dema_slow_4h = calculate_dema(close_4h, period=21)
    macd_hist_4h = calculate_macd(close_4h, fast=12, slow=26, signal=9)[2]
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
    ema_50_4h = calculate_ema(close_4h, period=50)
    
    # Align 4h indicators to 1h timeframe (auto shift for completed bars)
    dema_fast_4h_aligned = align_htf_to_ltf(prices, df_4h, dema_fast_4h)
    dema_slow_4h_aligned = align_htf_to_ltf(prices, df_4h, dema_slow_4h)
    macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries (wider range for 1h)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 25
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # MACD histogram threshold for momentum
    MACD_HIST_MIN = 0
    
    first_valid = max(200, 50, 26 + 9, 28, 14 * 2)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for NaN values
        if (np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(dema_fast_1h[i]) or
            np.isnan(dema_slow_1h[i]) or np.isnan(macd_hist_1h[i]) or np.isnan(adx_4h_aligned[i]) or
            atr_1h[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filters (HTF)
        trend_4h = 0
        if not np.isnan(dema_fast_4h_aligned[i]) and not np.isnan(dema_slow_4h_aligned[i]):
            if dema_fast_4h_aligned[i] > dema_slow_4h_aligned[i]:
                trend_4h = 1
            elif dema_fast_4h_aligned[i] < dema_slow_4h_aligned[i]:
                trend_4h = -1
        
        # 4h EMA50 filter
        ema_trend_4h = 0
        if not np.isnan(ema_50_4h_aligned[i]):
            if close[i] > ema_50_4h_aligned[i]:
                ema_trend_4h = 1
            elif close[i] < ema_50_4h_aligned[i]:
                ema_trend_4h = -1
        
        # 4h MACD momentum
        macd_momentum_4h = 0
        if not np.isnan(macd_hist_4h_aligned[i]):
            if macd_hist_4h_aligned[i] > MACD_HIST_MIN:
                macd_momentum_4h = 1
            elif macd_hist_4h_aligned[i] < -MACD_HIST_MIN:
                macd_momentum_4h = -1
        
        # 4h ADX filter - only trade when trend is strong enough
        adx_4h_val = adx_4h_aligned[i]
        if np.isnan(adx_4h_val) or adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 1h indicators for entry timing
        dema_crossover_1h = 0
        if dema_fast_1h[i] > dema_slow_1h[i]:
            dema_crossover_1h = 1
        elif dema_fast_1h[i] < dema_slow_1h[i]:
            dema_crossover_1h = -1
        
        macd_momentum_1h = 0
        if macd_hist_1h[i] > MACD_HIST_MIN:
            macd_momentum_1h = 1
        elif macd_hist_1h[i] < -MACD_HIST_MIN:
            macd_momentum_1h = -1
        
        # Donchian breakout confirmation
        donchian_signal_1h = 0
        if close[i] > donchian_upper_1h[i - 1] if i > 0 else 0:
            donchian_signal_1h = 1
        elif close[i] < donchian_lower_1h[i - 1] if i > 0 else 0:
            donchian_signal_1h = -1
        
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
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
        
        # Entry logic: 4h trend + 1h entry signals
        # Long entry: 4h bullish + 1h DEMA crossover + MACD + RSI pullback
        if (trend_4h == 1 and ema_trend_4h == 1 and macd_momentum_4h == 1 and
            dema_crossover_1h == 1 and macd_momentum_1h == 1 and
            RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
            signals[i] = SIZE_FULL
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Short entry: 4h bearish + 1h DEMA crossover + MACD + RSI pullback
        elif (trend_4h == -1 and ema_trend_4h == -1 and macd_momentum_4h == -1 and
              dema_crossover_1h == -1 and macd_momentum_1h == -1 and
              RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
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