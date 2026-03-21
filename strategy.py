#!/usr/bin/env python3
"""
EXPERIMENT #041 - MTF Donchian+KAMA+RSI+MACD+ATR (15m+1h+4h v1)
==================================================================================================
Hypothesis: Donchian breakouts on 4h provide strong trend direction (proven in #006 with +65.9%).
KAMA on 1h filters false breakouts adaptively. RSI on 15m provides optimal pullback entries.
MACD histogram confirms momentum before entry. ATR stoploss at 2.0*ATR for tight risk control.

Key changes from #040:
- Trend: 4h Donchian(20) breakout instead of HMA/Supertrend (more objective trend signal)
- Filter: 1h KAMA adaptive trend (worked in #006)
- Entry: 15m RSI(14) pullback to 40-60 range
- Momentum: 15m MACD histogram cross confirmation
- Stoploss: 2.0*ATR trailing (same as #040, proven effective)
- Position size: 0.35 (proven safe in winning strategies)
- Timeframe: 15m primary with 1h and 4h MTF filters

Why this should beat #040:
- Donchian breakouts are more objective than HMA crossovers
- 3-timeframe structure (4h trend, 1h filter, 15m entry) provides better confirmation
- MACD histogram adds momentum confirmation missing from #040
- Based on #006's success with Donchian+KAMA combination
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_kama_rsi_macd_atr_15m_v1"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    valid_macd = slow - 1
    signal_line[valid_macd + signal - 1] = np.mean(macd_line[valid_macd:valid_macd + signal])
    
    for i in range(valid_macd + signal, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    for i in range(valid_macd + signal - 1, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands and breakout signal)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    breakout = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        
        if i > period - 1:
            if high[i] > upper[i - 1]:
                breakout[i] = 1
            elif low[i] < lower[i - 1]:
                breakout[i] = -1
    
    return upper, lower, breakout


def resample_to_timeframe(prices, timeframe):
    """Resample prices to higher timeframe using proper aggregation"""
    prices_indexed = prices.set_index('open_time')
    
    if timeframe == '1h':
        df_resampled = prices_indexed.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    elif timeframe == '4h':
        df_resampled = prices_indexed.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    else:
        return prices
    
    return df_resampled


def align_mtf_indicator(base_index, mtf_values, mtf_index):
    """Align higher timeframe indicator back to base timeframe using ffill"""
    mtf_series = pd.Series(mtf_values, index=mtf_index)
    aligned = mtf_series.reindex(base_index, method='ffill').values
    return aligned


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    macd_15m, signal_15m, hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Resample to 1h for adaptive trend filter
    prices_df = prices.copy()
    prices_df['open_time'] = pd.to_datetime(prices_df['open_time'])
    prices_1h = resample_to_timeframe(prices_df, '1h')
    
    c_1h = prices_1h['close'].values
    h_1h = prices_1h['high'].values
    l_1h = prices_1h['low'].values
    idx_1h = prices_1h.index
    
    kama_1h = calculate_kama(c_1h, er_period=10, fast_period=2, slow_period=30)
    
    # Resample to 4h for Donchian trend
    prices_4h = resample_to_timeframe(prices_df, '4h')
    
    c_4h = prices_4h['close'].values
    h_4h = prices_4h['high'].values
    l_4h = prices_4h['low'].values
    idx_4h = prices_4h.index
    
    upper_4h, lower_4h, breakout_4h = calculate_donchian(h_4h, l_4h, period=20)
    
    # Align 1h and 4h indicators back to 15m timeframe
    base_index = pd.to_datetime(open_time)
    
    kama_1h_aligned = align_mtf_indicator(base_index, kama_1h, idx_1h)
    upper_4h_aligned = align_mtf_indicator(base_index, upper_4h, idx_4h)
    lower_4h_aligned = align_mtf_indicator(base_index, lower_4h, idx_4h)
    breakout_4h_aligned = align_mtf_indicator(base_index, breakout_4h, idx_4h)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # MACD histogram threshold for momentum confirmation
    MACD_HIST_MIN = 0.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Minimum warmup period
    first_valid = max(200, 40, 26 + 9, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(hist_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h Donchian trend direction
        donchian_trend = 0
        if upper_4h_aligned[i] > 0 and lower_4h_aligned[i] > 0:
            mid_4h = (upper_4h_aligned[i] + lower_4h_aligned[i]) / 2
            if close[i] > mid_4h:
                donchian_trend = 1
            elif close[i] < mid_4h:
                donchian_trend = -1
        
        # 1h KAMA adaptive trend
        kama_trend = 0
        if kama_1h_aligned[i] > 0:
            if close[i] > kama_1h_aligned[i]:
                kama_trend = 1
            elif close[i] < kama_1h_aligned[i]:
                kama_trend = -1
        
        # 15m RSI and MACD for entry timing
        rsi_val = rsi_15m[i]
        macd_hist = hist_15m[i]
        atr = atr_15m[i]
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
        
        # Entry logic: 4h Donchian + 1h KAMA + 15m RSI + MACD
        if donchian_trend == 1 and kama_trend == 1:  # Bullish trend confirmed
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                macd_hist > MACD_HIST_MIN):  # Pullback + momentum confirmation
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif donchian_trend == -1 and kama_trend == -1:  # Bearish trend confirmed
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                macd_hist < -MACD_HIST_MIN):  # Pullback + momentum confirmation
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