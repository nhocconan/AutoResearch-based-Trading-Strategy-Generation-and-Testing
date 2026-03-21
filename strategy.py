#!/usr/bin/env python3
"""
EXPERIMENT #008 - MTF HMA+Donchian+RSI+ADX (15m+1h+4h v1)
==================================================================================================
Hypothesis: Combine 4h HMA trend (fast trend filter) + 1h RSI momentum + 15m Donchian breakout 
entry + ADX strength filter. This differs from current best by:
- HMA instead of KAMA for faster trend response
- Donchian breakout instead of RSI pullback (breakout vs mean-reversion entry)
- ADX filter to avoid weak trends (new filter type not in current best)
- Three timeframes: 15m base, 1h momentum, 4h trend

Why this should work:
- 4h HMA provides clear trend direction with less lag than EMA
- Donchian breakout catches momentum moves early (vs waiting for RSI pullback)
- ADX > 25 filters out choppy markets (reduces whipsaws)
- 15m base timeframe has proven success in prior experiments
- Conservative position sizing (0.35 max) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_donchian_rsi_adx_15m_1h_4h_v1"
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    # Calculate TR
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Smooth with Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    # Initial values
    sum_plus_dm = np.sum(plus_dm[1:period+1])
    sum_minus_dm = np.sum(minus_dm[1:period+1])
    sum_tr = np.sum(tr[1:period+1])
    
    for i in range(period, n):
        if i == period:
            smoothed_plus_dm = sum_plus_dm
            smoothed_minus_dm = sum_minus_dm
            smoothed_tr = sum_tr
        else:
            smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / period + plus_dm[i]
            smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / period + minus_dm[i]
            smoothed_tr = smoothed_tr - smoothed_tr / period + tr[i]
        
        if smoothed_tr > 0:
            plus_di[i] = 100 * smoothed_plus_dm / smoothed_tr
            minus_di[i] = 100 * smoothed_minus_dm / smoothed_tr
        else:
            plus_di[i] = 0
            minus_di[i] = 0
        
        # Calculate DX
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period*2])
    
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper, lower, middle)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    
    return upper, middle, lower


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    adx_15m = calculate_adx(high, low, close, period=14)
    donchian_upper_15m, donchian_mid_15m, donchian_lower_15m = calculate_donchian(high, low, period=20)
    
    # Get 1h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_1h = get_htf_data(prices, '1h')
        c_1h = df_1h['close'].values
        h_1h = df_1h['high'].values
        l_1h = df_1h['low'].values
        
        # 1h indicators
        rsi_1h = calculate_rsi(c_1h, period=14)
        adx_1h = calculate_adx(h_1h, l_1h, c_1h, period=14)
        
        # Align 1h indicators to 15m timeframe (auto shift for completed bars)
        rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
        adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    except Exception:
        # Fallback if mtf_data fails
        rsi_1h_aligned = np.zeros(n)
        adx_1h_aligned = np.zeros(n)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(c_4h, period=21)
        adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        # Align 4h indicators to 15m timeframe
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        
        # Calculate 4h trend direction (price vs HMA)
        trend_4h = np.zeros(n)
        for i in range(n):
            if i < len(c_4h) and i < len(hma_4h_aligned):
                if c_4h[min(i // 16, len(c_4h) - 1)] > hma_4h_aligned[i]:
                    trend_4h[i] = 1
                elif c_4h[min(i // 16, len(c_4h) - 1)] < hma_4h_aligned[i]:
                    trend_4h[i] = -1
    except Exception:
        hma_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
        trend_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for momentum confirmation
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 70
    RSI_SHORT_MIN = 30
    RSI_SHORT_MAX = 55
    
    # ADX minimum for trend strength filter
    ADX_MIN = 25
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 14 * 3, 20, 26 + 9)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        trend_4h_val = trend_4h[i] if i < len(trend_4h) else 0
        rsi_1h = rsi_1h_aligned[i] if i < len(rsi_1h_aligned) else 50
        adx_1h = adx_1h_aligned[i] if i < len(adx_1h_aligned) else 0
        adx_4h = adx_4h_aligned[i] if i < len(adx_4h_aligned) else 0
        
        # ADX filter - avoid weak trends (both 1h and 4h must have strength)
        if adx_1h < ADX_MIN or adx_4h < ADX_MIN:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                position_side[i] = 0
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
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price) if prev_low > 0 else price
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
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
        
        # Entry logic: 4h trend + 1h RSI momentum + 15m Donchian breakout
        price = close[i]
        
        if trend_4h_val == 1:  # Bullish trend on 4h
            # RSI momentum on 1h (not overbought)
            # Donchian breakout on 15m (price breaks upper channel)
            if (RSI_LONG_MIN <= rsi_1h <= RSI_LONG_MAX and 
                price > donchian_upper_15m[i] and 
                donchian_upper_15m[i] > 0):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h_val == -1:  # Bearish trend on 4h
            # RSI momentum on 1h (not oversold)
            # Donchian breakout on 15m (price breaks lower channel)
            if (RSI_SHORT_MIN <= rsi_1h <= RSI_SHORT_MAX and 
                price < donchian_lower_15m[i] and 
                donchian_lower_15m[i] > 0):
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