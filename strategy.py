#!/usr/bin/env python3
"""
EXPERIMENT #023 - MTF HMA+RSI+ADX+ATR 15m+4h Clean v1
==================================================================================================
Hypothesis: Current best uses 15m+1h+4h but manual resampling violates mtf_data rules.
This version uses PROPER mtf_data helper for 4h trend + 15m entries.

Key changes from #040:
- USE mtf_data helper (get_htf_data, align_htf_to_ltf) - NO manual resampling!
- Timeframe: 15m entries + 4h trend (more stable than 1h)
- Position size: 0.30 (slightly more conservative than 0.35)
- Stoploss: 2.5*ATR (wider to avoid premature stops in crypto volatility)
- ADX threshold: 20 (lower to get more trades while maintaining quality)
- RSI thresholds: 35-65 (wider range for more entry opportunities)
- Add volatility filter using ATR percentile

Why this should beat #040:
- Proper MTF alignment eliminates look-ahead bias
- 4h trend is more stable than 1h (fewer whipsaws)
- Based on proven winning combinations from history
- Conservative sizing controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_adx_atr_15m_4h_v1"
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
    
    wma1 = pd.Series(close).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.sum(x * np.arange(1, half_period + 1)) / np.sum(np.arange(1, half_period + 1)),
        raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1)),
        raw=True
    ).values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, sqrt_period + 1)) / np.sum(np.arange(1, sqrt_period + 1)),
        raw=True
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return np.nan_to_num(rsi, nan=50.0)


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
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = atr > 0
    plus_di[mask] = 100 * plus_dm[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm[mask] / atr[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return np.nan_to_num(adx, nan=0.0)


def calculate_atr_percentile(atr, period=100):
    """Calculate ATR percentile for volatility regime filter"""
    n = len(atr)
    if n < period:
        return np.zeros(n)
    
    atr_pct = np.zeros(n)
    for i in range(period - 1, n):
        window = atr[i - period + 1:i + 1]
        rank = np.sum(window <= atr[i])
        atr_pct[i] = rank / period
    
    return atr_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    adx_15m = calculate_adx(high, low, close, period=14)
    atr_pct_15m = calculate_atr_percentile(atr_15m, period=100)
    
    # Get 4h data using mtf_data helper (CRITICAL - no manual resampling!)
    df_4h = get_htf_data(prices, '4h')
    
    if df_4h is None or len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h indicators on ACTUAL 4h data
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    hma_4h = calculate_hma(close_4h, period=48)
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
    
    # Align 4h indicators to 15m timeframe (auto shift by 1 for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries (wider range for more opportunities)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 20
    
    # ATR stoploss multiplier (wider for crypto volatility)
    ATR_STOP_MULT = 2.5
    
    # ATR percentile filter (avoid extreme volatility)
    ATR_PCT_MAX = 0.85
    ATR_PCT_MIN = 0.15
    
    first_valid = max(200, 100, 48, 14 * 2)
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend direction
        trend_4h = 0
        if close[i] > hma_4h_aligned[i]:
            trend_4h = 1
        elif close[i] < hma_4h_aligned[i]:
            trend_4h = -1
        
        adx_4h_val = adx_4h_aligned[i]
        atr_pct_val = atr_pct_15m[i]
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # ATR percentile filter - avoid extreme volatility regimes
        if atr_pct_val > ATR_PCT_MAX or atr_pct_val < ATR_PCT_MIN:
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
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
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
            
            # Hold position if no exit triggered and trend still valid
            if (prev_side == 1 and trend_4h == 1) or (prev_side == -1 and trend_4h == -1):
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                # Exit if trend reverses
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # Entry logic: 4h HMA trend + 4h ADX + 15m RSI pullback + ATR volatility filter
        if trend_4h == 1 and adx_4h_val >= ADX_MIN:  # Bullish trend confirmed on 4h
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):  # Pullback entry
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1 and adx_4h_val >= ADX_MIN:  # Bearish trend confirmed on 4h
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):  # Pullback entry
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