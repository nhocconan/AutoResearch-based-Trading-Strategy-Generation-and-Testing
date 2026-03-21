#!/usr/bin/env python3
"""
EXPERIMENT #006 - MTF KAMA+Donchian Trend + RSI Pullback + ADX Filter (4h+1h v1)
==================================================================================================
Hypothesis: Current best uses 4h DEMA+Supertrend. Let's try KAMA (Kaufman Adaptive) which adjusts
to market volatility - slower in choppy markets, faster in trending markets. Combine with Donchian
breakout confirmation on 4h, RSI pullback entries on 1h, and ADX filter to only trade strong trends.

Key changes from #005:
- Timeframe: 1h entries + 4h trend (proven MTF combo)
- Trend: KAMA(10,2,30) + Donchian(20) breakout on 4h
- Entry: RSI(14) pullback to 40-60 zone on 1h (proven entry method)
- Filter: ADX(14) > 25 for trend strength + Z-score(20) < 2.0
- Position size: 0.30 (discrete levels: 0.0, ±0.20, ±0.30)
- Stoploss: 2.0*ATR trailing, TP at 2R reduce to half

Why this should work:
- KAMA adapts to regime - reduces whipsaws in ranging markets
- Donchian breakout confirms trend direction
- RSI pullback entries catch retracements in trends (better than MACD for entries)
- ADX filter ensures we only trade when trend is strong (>25)
- 4h trend filter reduces false signals vs 1h-only strategies
"""

import numpy as np
import pandas as pd

name = "mtf_kama_donchian_rsi_adx_zscore_4h_1h_v1"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period - 1, n):
        if i >= slow_period - 1:
            change = abs(close[i] - close[i - slow_period + 1])
            volatility = np.sum(np.abs(np.diff(close[i - slow_period + 1:i + 1])))
            if volatility > 0:
                er[i] = change / volatility
            else:
                er[i] = 0
    
    # Calculate smoothing constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(period - 1, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    rsi = np.zeros(n)
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    for i in range(1, n):
        diff = close[i] - close[i - 1]
        gains[i] = max(0, diff)
        losses[i] = max(0, -diff)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gains[1:period + 1])
    avg_loss[period] = np.mean(losses[1:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    n = len(close)
    if n < period * 3:
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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


def resample_to_higher_tf(prices, tf='4h'):
    """Resample prices to higher timeframe using actual timestamps"""
    prices_indexed = prices.set_index('open_time')
    
    df_resampled = prices_indexed.resample(tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_resampled


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Resample to 4h for trend filters
    prices_df = prices.copy()
    prices_df['open_time'] = pd.to_datetime(prices_df['open_time'])
    
    try:
        df_4h = resample_to_higher_tf(prices_df, '4h')
    except Exception:
        # Fallback: simple downsampling if resample fails
        bars_per_4h = 4  # 4 x 1h = 4h
        n_4h = n // bars_per_4h
        
        c_4h = np.array([close[i * bars_per_4h + bars_per_4h - 1] for i in range(n_4h)])
        h_4h = np.array([np.max(high[i * bars_per_4h:i * bars_per_4h + bars_per_4h]) for i in range(n_4h)])
        l_4h = np.array([np.min(low[i * bars_per_4h:i * bars_per_4h + bars_per_4h]) for i in range(n_4h)])
        
        df_4h = pd.DataFrame({
            'open': c_4h,
            'high': h_4h,
            'low': l_4h,
            'close': c_4h,
            'volume': np.ones(n_4h)
        })
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    
    # 4h indicators for trend
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    n_4h = len(c_4h)
    
    kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Map 4h indicators back to 1h timeframe using ffill
    kama_trend_4h = np.zeros(n)
    donchian_trend_4h = np.zeros(n)
    adx_strength_4h = np.zeros(n)
    
    for i in range(n):
        # Find which 4h bar this 1h bar belongs to
        idx_4h = min(i // 4, n_4h - 1)
        if idx_4h >= 40:  # Need enough data for KAMA+ADX
            # KAMA trend: price above/below KAMA
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                kama_trend_4h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                kama_trend_4h[i] = -1
            
            # Donchian trend: price near upper/lower band
            price_range = donchian_upper_4h[idx_4h] - donchian_lower_4h[idx_4h]
            if price_range > 0:
                price_position = (c_4h[idx_4h] - donchian_lower_4h[idx_4h]) / price_range
                if price_position > 0.7:
                    donchian_trend_4h[i] = 1
                elif price_position < 0.3:
                    donchian_trend_4h[i] = -1
            
            adx_strength_4h[i] = adx_4h[idx_4h]
    
    # Entry thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    ZSCORE_MAX = 2.0
    ADX_MIN = 25
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 40 * 4, 14 + 1, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(zscore_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        kama_trend = kama_trend_4h[i]
        donchian_trend = donchian_trend_4h[i]
        adx_val = adx_strength_4h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
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
        
        # Entry logic: 4h KAMA + Donchian trend + ADX filter + 1h RSI pullback + Z-score
        if kama_trend == 1 and donchian_trend == 1 and adx_val > ADX_MIN:  # Bullish trend confirmed
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and
                abs(zscore_val) < ZSCORE_MAX):  # RSI pullback + not extreme
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif kama_trend == -1 and donchian_trend == -1 and adx_val > ADX_MIN:  # Bearish trend confirmed
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and
                abs(zscore_val) < ZSCORE_MAX):  # RSI pullback + not extreme
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