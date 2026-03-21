#!/usr/bin/env python3
"""
EXPERIMENT #010 - MTF EMA+Bollinger+RSI+ATR (15m+1h+4h v1)
==================================================================================================
Hypothesis: 4h EMA(21/55) crossover provides clean trend direction. 1h Bollinger %B identifies
pullback entries within the trend (buy when %B < 0.3 in uptrend, sell when %B > 0.7 in downtrend).
15m RSI confirms entry timing. ATR stoploss at 2.0*ATR for tight risk control.

Key changes from #009:
- Trend: 4h EMA(21/55) crossover instead of Donchian (smoother, less whipsaw)
- Entry filter: 1h Bollinger %B for pullback detection (mean reversion within trend)
- Entry timing: 15m RSI(14) for precise entry
- Stoploss: 2.0*ATR trailing (proven effective)
- Position size: 0.35 max (discrete levels: 0.0, ±0.25, ±0.35)
- Timeframe: 15m primary with 1h and 4h MTF filters
- FIX: Use integer index mapping instead of datetime comparison to avoid timezone issues

Why this should beat #005 (Sharpe=0.213):
- EMA crossovers are smoother than Donchian breakouts (fewer false signals)
- Bollinger %B adds mean-reversion entry logic (buy dips in uptrend)
- 3-timeframe structure provides layered confirmation
- Based on classic trend-following with pullback entry (proven edge)
"""

import numpy as np
import pandas as pd

name = "mtf_ema_bollinger_rsi_atr_15m_v1"
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
    ema[period - 1] = np.mean(close[:period])
    
    multiplier = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = ema[i - 1] + multiplier * (close[i] - ema[i - 1])
    
    return ema


def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and %B"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    pct_b = np.zeros(n)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = sma[i] + std_mult * std
        lower[i] = sma[i] - std_mult * std
        
        if upper[i] != lower[i]:
            pct_b[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
        else:
            pct_b[i] = 0.5
    
    return upper, lower, pct_b


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


def resample_to_timeframe(prices, timeframe):
    """Resample prices to higher timeframe using proper aggregation"""
    prices_indexed = prices.set_index('open_time')
    prices_indexed.index = pd.to_datetime(prices_indexed.index, utc=True)
    
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Initialize output arrays
    signals = np.zeros(n)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    
    # Resample to 1h for Bollinger filter
    prices_df = prices.copy()
    prices_df['open_time'] = pd.to_datetime(prices_df['open_time'], utc=True)
    prices_1h = resample_to_timeframe(prices_df, '1h')
    
    c_1h = prices_1h['close'].values
    upper_1h, lower_1h, pct_b_1h = calculate_bollinger(c_1h, period=20, std_mult=2.0)
    
    # Resample to 4h for EMA trend
    prices_4h = resample_to_timeframe(prices_df, '4h')
    
    c_4h = prices_4h['close'].values
    ema21_4h = calculate_ema(c_4h, period=21)
    ema55_4h = calculate_ema(c_4h, period=55)
    
    # Create mapping from 15m index to 1h and 4h indices
    # Use integer-based alignment to avoid datetime timezone issues
    base_times = pd.to_datetime(prices['open_time'], utc=True)
    
    # Map 1h times to 15m indices
    times_1h = prices_1h.index
    times_4h = prices_4h.index
    
    # Create forward-fill mapping arrays
    idx_1h_aligned = np.zeros(n, dtype=int)
    idx_4h_aligned = np.zeros(n, dtype=int)
    
    # For each 15m bar, find the most recent 1h and 4h bar
    j_1h = 0
    j_4h = 0
    for i in range(n):
        # Find latest 1h bar <= current 15m time
        while j_1h < len(times_1h) - 1 and times_1h[j_1h + 1] <= base_times[i]:
            j_1h += 1
        idx_1h_aligned[i] = j_1h
        
        # Find latest 4h bar <= current 15m time
        while j_4h < len(times_4h) - 1 and times_4h[j_4h + 1] <= base_times[i]:
            j_4h += 1
        idx_4h_aligned[i] = j_4h
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Minimum warmup period
    first_valid = max(200, 55, 40, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for valid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned indices
        idx_1h = idx_1h_aligned[i]
        idx_4h = idx_4h_aligned[i]
        
        # Check bounds for MTF data
        if idx_1h >= len(pct_b_1h) or idx_4h >= len(ema21_4h):
            signals[i] = 0.0
            continue
        
        # 4h EMA trend direction
        ema_trend = 0
        if ema21_4h[idx_4h] > 0 and ema55_4h[idx_4h] > 0:
            if ema21_4h[idx_4h] > ema55_4h[idx_4h]:
                ema_trend = 1  # Bullish
            elif ema21_4h[idx_4h] < ema55_4h[idx_4h]:
                ema_trend = -1  # Bearish
        
        # 1h Bollinger %B for pullback detection
        pct_b = pct_b_1h[idx_1h]
        
        # 15m RSI for entry timing
        rsi_val = rsi_15m[i]
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
        
        # Entry logic: 4h EMA trend + 1h Bollinger pullback + 15m RSI timing
        # Long: EMA bullish + %B < 0.4 (pullback) + RSI 35-55 (not overbought)
        if ema_trend == 1 and pct_b < 0.4 and 35 <= rsi_val <= 55:
            signals[i] = SIZE_FULL
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Short: EMA bearish + %B > 0.6 (rally) + RSI 45-65 (not oversold)
        elif ema_trend == -1 and pct_b > 0.6 and 45 <= rsi_val <= 65:
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