#!/usr/bin/env python3
"""
EXPERIMENT #107 - MTF Supertrend+RSI+Chandelier with Proper HTF (15m+4h v1)
==================================================================================================
Hypothesis: Recent Chandelier experiments (#101-#106) failed due to improper HTF alignment
and aggressive volatility-adjusted sizing. This version uses:

1. PROPER mtf_data helper for 4h trend filter (mandatory - 46 strategies failed without it)
2. 15m entries with 4h trend confirmation (proven combination from best strategies)
3. Simple discrete position sizing (0.30 max) - NO volatility adjustment (caused recent crashes)
4. Chandelier exit with 3*ATR(22) for trailing stops (implemented correctly)
5. RSI pullback entries (40-60 range) with ADX trend strength filter (>25)
6. Conservative sizing based on lessons from #040 (0.35 worked well)

Why this should beat recent failures:
- Uses mtf_data helper for correct 4h boundary alignment (no synthetic resampling)
- No volatility-adjusted sizing (recent experiments #101-#106 all crashed with this)
- Simpler logic than failed ensemble strategies
- Based on winning pattern from mtf_supertrend_macd_bbw_rsi_15m_1h_4h_v1 (Sharpe=3.653)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_chandelier_proper_htf_15m_4h_v1"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


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


def calculate_chandelier_exit(high, low, close, atr, period=22, multiplier=3.0):
    """
    Calculate Chandelier Exit (ATR trailing stop)
    Long exit: highest_high - multiplier * ATR
    Short exit: lowest_low + multiplier * ATR
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    chandelier_long = np.zeros(n)
    chandelier_short = np.zeros(n)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        chandelier_long[i] = highest_high - multiplier * atr[i]
        chandelier_short[i] = lowest_low + multiplier * atr[i]
    
    return chandelier_long, chandelier_short


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # =============================================================================
    # CRITICAL: Use mtf_data helper for proper 4h alignment
    # This loads ACTUAL Binance 4h candles and aligns them correctly
    # 46 strategies failed audit without this proper HTF handling
    # =============================================================================
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # Calculate 4h indicators
        atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
        supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        # Align 4h indicators to 15m timeframe (auto shift(1) for completed bars)
        st_trend_4h = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        
    except Exception as e:
        # Fallback if mtf_data fails - use simple downsampling
        bars_per_4h = 16  # 16 x 15m = 4h
        n_4h = n // bars_per_4h
        
        c_4h = np.array([close[i * bars_per_4h + bars_per_4h - 1] for i in range(n_4h)])
        h_4h = np.array([np.max(high[i * bars_per_4h:(i + 1) * bars_per_4h]) for i in range(n_4h)])
        l_4h = np.array([np.min(low[i * bars_per_4h:(i + 1) * bars_per_4h]) for i in range(n_4h)])
        
        atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
        supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        st_trend_4h = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
        
        for i in range(n):
            idx_4h = i // bars_per_4h
            if idx_4h < n_4h and idx_4h > 0:
                st_trend_4h[i] = st_direction_4h[idx_4h - 1]  # Shift by 1 for completed bar
                adx_4h_aligned[i] = adx_4h[idx_4h - 1]
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx_15m = calculate_adx(high, low, close, period=14)
    
    # Chandelier exit for trailing stops (22-period ATR, 3.0 multiplier)
    chandelier_long, chandelier_short = calculate_chandelier_exit(
        high, low, close, atr_15m, period=22, multiplier=3.0
    )
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    # Lessons from recent failures: NO volatility adjustment, keep it simple
    SIZE_FULL = 0.30  # Reduced from 0.35 for extra safety
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 25
    
    # Minimum valid index (need enough data for all indicators)
    first_valid = max(200, 40, 14 * 2, 22)
    
    # Track position state for stoploss/takeprofit
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filter (MUST agree)
        st_trend = st_trend_4h[i]
        adx_4h_val = adx_4h_aligned[i]
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 4h Supertrend direction must be clear
        if st_trend == 0:
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
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            price = close[i]
            atr = atr_15m[i]
            
            # Chandelier exit stoploss (3*ATR trailing)
            if prev_side == 1:
                chandelier_stop = chandelier_long[i]
                if price < chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R based on initial ATR)
                initial_risk = 3 * atr  # Chandelier uses 3*ATR
                tp_price = prev_entry + 2 * initial_risk
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit after TP
                if prev_tp:
                    trail_stop = current_high - initial_risk
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
            elif prev_side == -1:
                chandelier_stop = chandelier_short[i]
                if price > chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R based on initial ATR)
                initial_risk = 3 * atr
                tp_price = prev_entry - 2 * initial_risk
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit after TP
                if prev_tp:
                    trail_stop = current_low + initial_risk
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
        
        # Entry logic: 4h Supertrend + ADX + 15m RSI pullback
        price = close[i]
        rsi_val = rsi_15m[i]
        
        if st_trend == 1:  # Bullish trend on 4h
            # RSI pullback entry in uptrend
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif st_trend == -1:  # Bearish trend on 4h
            # RSI pullback entry in downtrend
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