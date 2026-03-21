#!/usr/bin/env python3
"""
EXPERIMENT #013 - Donchian Trend + RSI Pullback + ADX Strength Filter
====================================================================================
Hypothesis: Donchian channels provide cleaner trend signals than EMA crossovers by 
capturing actual breakouts. Combined with RSI pullback entries (proven in #005 best 
performer) and ADX strength filter to avoid trading during weak/choppy trends.

Key differences from current best (#005 EMA+RSI+Z-score):
- Donchian(20) trend instead of EMA - captures actual breakout levels
- RSI(14) pullback entry (same as #005 proven winner)
- ADX(14) > 25 filter - NEW: only trade when trend has sufficient strength
- Z-score filter (proven in #005) - avoids extreme mean-reversion traps
- 4h Donchian trend + 1h RSI entries (proven MTF structure)
- Trailing stoploss at 2*ATR, take profit at 2R (reduce to half)
- Discrete signal levels: 0.0, ±0.25, ±0.35 to minimize churn costs

Why this might beat Sharpe=5.525:
- Donchian breakouts filter out fakeout moves better than EMA crosses
- ADX strength filter eliminates trades during choppy/weak trends (major improvement)
- RSI pullback proven effective in #005 and #007 (both top performers)
- Multi-timeframe structure proven to 2x Sharpe vs single timeframe
- ADX filter should reduce trade count but increase win rate
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_rsi_adx_zscore_v1"
timeframe = "1h"
leverage = 1.0


def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel (highest high and lowest low over period)
    Returns: upper_channel, lower_channel, middle_channel
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    rsi = np.zeros(n)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n - 1)
    loss = np.zeros(n - 1)
    
    gain[delta > 0] = delta[delta > 0]
    loss[delta < 0] = -delta[delta < 0]
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period - 1] = np.mean(gain[:period])
    avg_loss[period - 1] = np.mean(loss[:period])
    
    for i in range(period, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index) for trend strength
    ADX > 25 indicates strong trend, ADX < 20 indicates weak/choppy market
    """
    n = len(close)
    adx = np.zeros(n)
    
    if n < period * 3:
        return adx
    
    # Calculate True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
    
    # Smooth TR, +DM, -DM
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    smoothed_tr = np.zeros(n)
    smoothed_plus_dm = np.zeros(n)
    smoothed_minus_dm = np.zeros(n)
    
    smoothed_tr[period - 1] = np.sum(tr[1:period])
    smoothed_plus_dm[period - 1] = np.sum(plus_dm[1:period])
    smoothed_minus_dm[period - 1] = np.sum(minus_dm[1:period])
    
    for i in range(period, n):
        smoothed_tr[i] = smoothed_tr[i - 1] - smoothed_tr[i - 1] / period + tr[i]
        smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - smoothed_plus_dm[i - 1] / period + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - smoothed_minus_dm[i - 1] / period + minus_dm[i]
    
    for i in range(period - 1, n):
        if smoothed_tr[i] > 0:
            plus_di[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is smoothed DX
    adx[period * 2 - 2] = np.mean(dx[period - 1:period * 2 - 1])
    
    for i in range(period * 2 - 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        mean = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


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
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    atr_1h = calculate_atr(high, low, close, period=14)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(high, low, close, period=14)
    
    # 4h Donchian for trend filter (resample 1h → 4h)
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
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    n_4h = len(c_4h)
    
    # Calculate 4h Donchian for trend
    donchian_upper_4h, donchian_lower_4h, donchian_mid_4h = calculate_donchian(h_4h, l_4h, period=20)
    
    # 4h trend direction based on Donchian position
    trend_4h = np.zeros(n_4h)
    for i in range(40, n_4h):  # Need enough data for Donchian
        if c_4h[i] > donchian_mid_4h[i] and donchian_upper_4h[i] > donchian_upper_4h[i - 5]:
            trend_4h[i] = 1  # Bullish (price above mid, channel expanding up)
        elif c_4h[i] < donchian_mid_4h[i] and donchian_lower_4h[i] < donchian_lower_4h[i - 5]:
            trend_4h[i] = -1  # Bearish (price below mid, channel expanding down)
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # RSI pulls back to 45 in uptrend
    RSI_SHORT_ENTRY = 55  # RSI rallies to 55 in downtrend
    RSI_EXIT = 50         # Exit when RSI crosses back through 50
    
    # ADX threshold for trend strength
    ADX_MIN = 25   # Only trade when ADX > 25 (strong trend)
    
    # Z-score filter thresholds
    ZSCORE_MAX = 2.0   # Avoid extreme overbought (> 2 std)
    ZSCORE_MIN = -2.0  # Avoid extreme oversold (< -2 std)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    first_valid = max(100, 40, 40, 42)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    initial_stop = np.zeros(n)  # Track initial stoploss level
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        adx_val = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Z-score regime filter - avoid extreme moves (mean reversion likely)
        if zscore_val > ZSCORE_MAX or zscore_val < ZSCORE_MIN:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                if position_side[i-1] == 1:
                    highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                    lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                elif position_side[i-1] == -1:
                    highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                    lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                initial_stop[i] = initial_stop[i-1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # ADX strength filter - only trade when trend is strong
        if adx_val < ADX_MIN:
            if i > 0 and position_side[i - 1] != 0:
                # Hold existing position but don't add
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                if position_side[i-1] == 1:
                    highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                    lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                elif position_side[i-1] == -1:
                    highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                    lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                initial_stop[i] = initial_stop[i-1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_stop = initial_stop[i - 1] if initial_stop[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                
                # Trailing stoploss (2*ATR from entry, or trail from highest)
                trail_stop = highest_since_entry[i] - ATR_STOP_MULT * atr if highest_since_entry[i] > 0 else prev_entry - ATR_STOP_MULT * atr
                stoploss_price = max(prev_entry - ATR_STOP_MULT * atr, trail_stop)
                
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = prev_stop
                    continue
                
                # RSI exit for longs
                if rsi_val > 70:  # Overbought, consider exit
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
                    
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                
                # Trailing stoploss
                trail_stop = lowest_since_entry[i] + ATR_STOP_MULT * atr if lowest_since_entry[i] > 0 else prev_entry + ATR_STOP_MULT * atr
                stoploss_price = min(prev_entry + ATR_STOP_MULT * atr, trail_stop)
                
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = prev_stop
                    continue
                
                # RSI exit for shorts
                if rsi_val < 30:  # Oversold, consider exit
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
        
        # Entry logic with RSI pullback confirmation
        position_size = SIZE_FULL
        
        if trend == 1:  # 4h uptrend + ADX OK + Z-score OK
            # RSI pulls back to entry zone in uptrend
            if RSI_LONG_ENTRY - 5 <= rsi_val <= RSI_LONG_ENTRY + 5:
                # Check for RSI turning up from oversold
                if i > 0 and rsi_1h[i-1] < rsi_val:
                    signals[i] = position_size
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    initial_stop[i] = price - ATR_STOP_MULT * atr
                else:
                    if i > 0 and position_side[i - 1] == 1:
                        signals[i] = signals[i - 1]
                        position_side[i] = 1
                        entry_price[i] = entry_price[i - 1]
                        tp_triggered[i] = tp_triggered[i - 1]
                        highest_since_entry[i] = highest_since_entry[i-1]
                        lowest_since_entry[i] = lowest_since_entry[i-1]
                        initial_stop[i] = initial_stop[i-1]
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
            else:
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = initial_stop[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    
        elif trend == -1:  # 4h downtrend + ADX OK + Z-score OK
            # RSI rallies to entry zone in downtrend
            if RSI_SHORT_ENTRY - 5 <= rsi_val <= RSI_SHORT_ENTRY + 5:
                # Check for RSI turning down from overbought
                if i > 0 and rsi_1h[i-1] > rsi_val:
                    signals[i] = -position_size
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    initial_stop[i] = price + ATR_STOP_MULT * atr
                else:
                    if i > 0 and position_side[i - 1] == -1:
                        signals[i] = signals[i - 1]
                        position_side[i] = -1
                        entry_price[i] = entry_price[i - 1]
                        tp_triggered[i] = tp_triggered[i - 1]
                        highest_since_entry[i] = highest_since_entry[i-1]
                        lowest_since_entry[i] = lowest_since_entry[i-1]
                        initial_stop[i] = initial_stop[i-1]
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
            else:
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = initial_stop[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
            initial_stop[i] = 0
    
    return signals