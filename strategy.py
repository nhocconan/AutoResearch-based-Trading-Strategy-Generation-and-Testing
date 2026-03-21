#!/usr/bin/env python3
"""
EXPERIMENT #031 - MTF Supertrend+Zscore+RSI (1h+4h Dynamic Sizing)
==================================================================================================
Hypothesis: Based on #030 (Sharpe=5.787), switching to 1h timeframe reduces trading costs while 4h trend filter remains effective. Adding Z-score for mean reversion entries within trend and Supertrend for cleaner signals should improve risk-adjusted returns. Dynamic ATR-based position sizing controls drawdown better during volatile periods.

Key changes from #030:
- Timeframe: 1h (fewer trades = less fees, more stable signals)
- MTF: 1h + 4h (proven combination from #026)
- Supertrend instead of HMA/KAMA (cleaner trend signals, less lag)
- Z-score filter for mean reversion entries within trend
- Dynamic position sizing: base_size * (target_vol / current_vol)
- Tighter stoploss: 2.0*ATR (vs 2.5*ATR in #030)
- Position size: 0.30 base (vs 0.35 in #030)

Why this should beat #030:
- 1h has fewer false signals than 15m
- Supertrend is more responsive than HMA/KAMA
- Z-score adds mean reversion edge within trend
- Dynamic sizing reduces risk during high volatility
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_zscore_rsi_1h_4h_v1"
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


def calculate_supertrend(high, low, close, atr, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper = mid + multiplier * atr[i]
        lower = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper
            direction[i] = 1
        else:
            if close[i - 1] > supertrend[i - 1]:
                supertrend[i] = min(upper, supertrend[i - 1])
                direction[i] = 1
            else:
                supertrend[i] = max(lower, supertrend[i - 1])
                direction[i] = -1
    
    return supertrend, direction


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion"""
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    signals = np.zeros(n)
    
    # Check if open_time exists for proper resampling
    if 'open_time' in prices.columns:
        prices_indexed = prices.set_index('open_time')
        
        # Resample to 4h for trend filter
        df_4h = prices_indexed.resample('4h').agg({
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
        
        # 4h ATR and Supertrend for trend direction
        atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
        _, direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, atr_4h, period=10, multiplier=3.0)
        
        # Map 4h trend to 1h
        trend_4h_series = pd.Series(direction_4h, index=df_4h.index)
        trend_4h = trend_4h_series.reindex(prices_indexed.index, method='ffill').fillna(0).values
    else:
        # Fallback
        trend_4h = np.zeros(n)
    
    # 1h indicators
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    
    # Position sizing parameters
    BASE_SIZE = 0.30
    TARGET_VOL = 0.02  # 2% daily volatility target
    
    # Entry thresholds
    ZSCORE_LONG_MAX = -0.5  # Pullback entry (price below mean)
    ZSCORE_SHORT_MIN = 0.5  # Rally entry (price above mean)
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    
    # Stoploss
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 40 * 4, 14 * 2, 20)
    
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
        
        trend = trend_4h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Dynamic position sizing based on volatility
        current_vol = atr / price if price > 0 else 0.01
        vol_adjustment = min(1.5, max(0.5, TARGET_VOL / current_vol)) if current_vol > 0 else 1.0
        position_size = BASE_SIZE * vol_adjustment
        position_size = min(0.40, max(0.15, position_size))  # Clamp between 0.15 and 0.40
        
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
                    signals[i] = position_size / 2
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
                    signals[i] = -position_size / 2
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
        
        # Entry logic: 4h Supertrend + 1h RSI + Z-score
        if trend == 1:  # Bullish trend on 4h
            if (zscore_val <= ZSCORE_LONG_MAX and  # Mean reversion pullback
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):  # RSI in healthy range
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1:  # Bearish trend on 4h
            if (zscore_val >= ZSCORE_SHORT_MIN and  # Mean reversion rally
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):  # RSI in healthy range
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals