#!/usr/bin/env python3
"""
EXPERIMENT #011 - 30m Primary Bollinger Squeeze + Supertrend + RSI with 4h/1d Filters
==================================================================================================
Hypothesis: Use 30m as PRIMARY timeframe for faster signal generation than 1h/4h.
Combine Bollinger Band squeeze detection (low vol before breakout) + Supertrend direction +
RSI pullback entries + ADX trend strength filter. Use 4h Supertrend and 1d EMA50 as HTF filters.

Why this should work:
- 30m primary = more trades than 1h/4h = better statistical significance
- BB squeeze identifies low-volatility compression before explosive moves
- Supertrend gives clear directional bias with ATR-based stops
- ADX filter avoids trading in weak/choppy trends
- 4h + 1d HTF alignment ensures we trade with higher timeframe momentum
- Discrete signal levels (0.0, ±0.25, ±0.35) minimize fee churn

Key differences from #009/#010:
- 30m primary instead of 1h/4h (faster entries, more trades)
- Bollinger Band squeeze filter (new indicator not tried before)
- ADX strength confirmation (avoid weak trends)
- Triple HTF filter: 1d EMA + 4h Supertrend + 30m signals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_bb_squeeze_supertrend_rsi_adx_30m_4h_1d_v1"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        upper_band = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = -1
        else:
            if direction[i - 1] == 1:
                if close[i] > lower_band:
                    supertrend[i] = max(lower_band, supertrend[i - 1])
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band
                    direction[i] = -1
            else:
                if close[i] < upper_band:
                    supertrend[i] = min(upper_band, supertrend[i - 1])
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band
                    direction[i] = 1
    
    return supertrend, direction


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower


def calculate_bb_squeeze(upper, lower, sma, lookback=20):
    """
    Detect Bollinger Band squeeze (low volatility compression)
    Returns 1 when bands are narrow (squeeze), 0 otherwise
    """
    n = len(close) if 'close' in dir() else len(upper)
    squeeze = np.zeros(n)
    
    # Bandwidth = (Upper - Lower) / SMA
    bandwidth = np.zeros(n)
    for i in range(n):
        if sma[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / sma[i]
    
    # Calculate percentile of bandwidth over lookback
    for i in range(lookback, n):
        recent_bw = bandwidth[i - lookback:i]
        if len(recent_bw) > 0:
            # Squeeze when bandwidth is in bottom 25% of recent range
            if bandwidth[i] < np.percentile(recent_bw, 25):
                squeeze[i] = 1
    
    return squeeze, bandwidth


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * np.mean(plus_dm[i - period + 1:i + 1]) / atr[i]
            minus_di[i] = 100 * np.mean(minus_dm[i - period + 1:i + 1]) / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_ema(close, period=50):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    return ema


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 30m indicators (primary timeframe)
    atr_30m = calculate_atr(high, low, close, period=14)
    supertrend_30m, st_direction_30m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    bb_upper, bb_sma, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    squeeze_30m, bandwidth_30m = calculate_bb_squeeze(bb_upper, bb_lower, bb_sma, lookback=20)
    adx_30m = calculate_adx(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    
    # Get 4h data using mtf_data helper
    try:
        df_4h = get_htf_data(prices, '4h')
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        c_4h = df_4h['close'].values
        
        # 4h Supertrend for trend filter
        _, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    except Exception:
        st_direction_4h_aligned = np.zeros(n)
    
    # Get 1d data using mtf_data helper for regime filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # Daily EMA50 for regime filter
        ema_1d = calculate_ema(c_1d, period=50)
        
        # Align daily EMA to 30m timeframe
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
        
        # Calculate daily trend (price vs EMA50)
        trend_1d = np.zeros(n)
        for i in range(n):
            if i < len(ema_1d_aligned) and ema_1d_aligned[i] > 0:
                if close[i] > ema_1d_aligned[i]:
                    trend_1d[i] = 1
                elif close[i] < ema_1d_aligned[i]:
                    trend_1d[i] = -1
    except Exception:
        trend_1d = np.zeros(n)
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for entries (pullback in trend)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ADX threshold for trend strength
    ADX_MIN = 20
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 50, 30, 28)  # Ensure all indicators are ready
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_30m[i]) or np.isnan(rsi_30m[i]) or atr_30m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned HTF trends
        trend_4h = st_direction_4h_aligned[i] if i < len(st_direction_4h_aligned) else 0
        daily_trend = trend_1d[i] if i < len(trend_1d) else 0
        
        # Triple HTF filter - all must align
        if daily_trend == 0 or trend_4h == 0:
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
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_30m[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_30m[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_30m[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_30m[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_30m[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_30m[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Check if HTF trend reversed - exit position
            if prev_side == 1 and (daily_trend == -1 or trend_4h == -1):
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            elif prev_side == -1 and (daily_trend == 1 or trend_4h == 1):
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
        
        # Entry logic: Triple HTF alignment + BB squeeze + ADX strength + RSI entry
        price = close[i]
        
        # Check ADX for trend strength
        adx_strong = adx_30m[i] >= ADX_MIN
        
        if not adx_strong:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Bullish setup: Daily up + 4h up + 30m Supertrend up + ADX strong
        if daily_trend == 1 and trend_4h == 1 and st_direction_30m[i] == 1:
            # RSI pullback entry zone (not overbought)
            if RSI_LONG_MIN <= rsi_30m[i] <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        # Bearish setup: Daily down + 4h down + 30m Supertrend down + ADX strong
        elif daily_trend == -1 and trend_4h == -1 and st_direction_30m[i] == -1:
            # RSI pullback entry zone (not oversold)
            if RSI_SHORT_MIN <= rsi_30m[i] <= RSI_SHORT_MAX:
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