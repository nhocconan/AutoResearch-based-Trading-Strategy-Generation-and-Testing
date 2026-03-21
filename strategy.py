#!/usr/bin/env python3
"""
EXPERIMENT #032 - MTF HMA+KAMA+Stoch+RSI+BBW (15m+4h Optimized)
==================================================================================================
Hypothesis: #030 achieved Sharpe=5.787 using 15m+4h with HMA/KAMA trend + Stoch/RSI entries + BBW regime.
#031 failed (Sharpe=0.35) by switching to 1h (too few trades) and Supertrend (less effective than HMA/KAMA).

Key changes from #030:
- Return to 15m timeframe (proven optimal in #030)
- HMA(16/48) + KAMA(ER=10) trend combination
- 4h trend filter using HMA slope
- Stochastic(14,3,3) + RSI(14) for entry timing
- Bollinger Band Width percentile for regime detection
- Discrete signal levels: 0.0, ±0.25, ±0.35 (reduce churn costs)
- Dynamic ATR-based position sizing: base * (target_vol / current_vol)
- Stoploss: 2.0*ATR trailing
- Take profit: 2R with trail at 1R

Why this should beat #030:
- Refined entry thresholds based on #030 success
- Better position sizing clamp (0.20-0.35 range)
- Cleaner multi-timeframe alignment
- Reduced signal churn with discrete levels
"""

import numpy as np
import pandas as pd

name = "mtf_hma_kama_stoch_rsi_bbw_15m_4h_v3"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period * 3:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    # WMA with period/2
    for i in range(half - 1, n):
        weights = np.arange(1, half + 1)
        window = close[i - half + 1:i + 1]
        wma1[i] = np.sum(window * weights) / np.sum(weights)
    
    # WMA with period
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        window = close[i - period + 1:i + 1]
        wma2[i] = np.sum(window * weights) / np.sum(weights)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    raw_hma = 2 * wma1 - wma2
    
    for i in range(sqrt_period - 1, n):
        weights = np.arange(1, sqrt_period + 1)
        window = raw_hma[i - sqrt_period + 1:i + 1]
        if np.sum(weights) > 0:
            hma[i] = np.sum(window * weights) / np.sum(weights)
    
    return hma


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        # Efficiency Ratio
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        # Smoothing constant
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    n = len(close)
    if n < k_period + d_period:
        return np.zeros(n), np.zeros(n)
    
    stoch_k = np.zeros(n)
    stoch_d = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        
        if highest_high - lowest_low > 0:
            stoch_k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            stoch_k[i] = 50
    
    # %D is SMA of %K
    for i in range(k_period + d_period - 2, n):
        stoch_d[i] = np.mean(stoch_k[i - d_period + 1:i + 1])
    
    return stoch_k, stoch_d


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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    bb_mid = np.zeros(n)
    bb_upper = np.zeros(n)
    bb_lower = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        bb_mid[i] = np.mean(window)
        std = np.std(window)
        bb_upper[i] = bb_mid[i] + std_mult * std
        bb_lower[i] = bb_mid[i] - std_mult * std
    
    return bb_mid, bb_upper, bb_lower


def calculate_bbw_percentile(bb_upper, bb_lower, bb_mid, lookback=100):
    """Calculate Bollinger Band Width percentile for regime detection"""
    n = len(bb_mid)
    bbw_pct = np.zeros(n)
    
    bbw = np.zeros(n)
    for i in range(n):
        if bb_mid[i] > 0:
            bbw[i] = (bb_upper[i] - bb_lower[i]) / bb_mid[i]
        else:
            bbw[i] = 0
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        bbw_pct[i] = np.sum(bbw[i] >= window) / len(window)
    
    return bbw_pct


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
        n_4h = len(c_4h)
        
        # 4h HMA for trend direction
        hma_4h_fast = calculate_hma(c_4h, period=16)
        hma_4h_slow = calculate_hma(c_4h, period=48)
        
        # 4h trend: fast HMA > slow HMA = bullish
        trend_4h = np.zeros(n_4h)
        for i in range(len(c_4h)):
            if hma_4h_fast[i] > hma_4h_slow[i] and hma_4h_fast[i] > 0:
                trend_4h[i] = 1
            elif hma_4h_fast[i] < hma_4h_slow[i] and hma_4h_fast[i] > 0:
                trend_4h[i] = -1
            else:
                trend_4h[i] = 0
        
        # Map 4h trend to 15m
        trend_4h_series = pd.Series(trend_4h, index=df_4h.index)
        trend_4h_aligned = trend_4h_series.reindex(prices_indexed.index, method='ffill').fillna(0).values
    else:
        trend_4h_aligned = np.zeros(n)
    
    # 15m indicators
    hma_15m_fast = calculate_hma(close, period=16)
    hma_15m_slow = calculate_hma(close, period=48)
    kama_15m = calculate_kama(close, period=10)
    atr_15m = calculate_atr(high, low, close, period=14)
    stoch_k, stoch_d = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    rsi_15m = calculate_rsi(close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct = calculate_bbw_percentile(bb_upper, bb_lower, bb_mid, lookback=100)
    
    # Position sizing parameters
    BASE_SIZE = 0.30
    TARGET_VOL = 0.015  # 1.5% daily volatility target
    MIN_SIZE = 0.20
    MAX_SIZE = 0.35
    
    # Entry thresholds (optimized from #030)
    STOCH_LONG_MAX = 50  # Stoch %K below 50 for long entry
    STOCH_SHORT_MIN = 50  # Stoch %K above 50 for short entry
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    BBW_REGIME_MIN = 0.3  # Not in extreme squeeze
    BBW_REGIME_MAX = 0.8  # Not in extreme expansion
    
    # Stoploss
    ATR_STOP_MULT = 2.0
    
    first_valid = max(150, 48 * 4, 14 * 2, 20, 100)
    
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
        
        trend = trend_4h_aligned[i]
        rsi_val = rsi_15m[i]
        stoch_val = stoch_k[i]
        bbw_val = bbw_pct[i]
        atr = atr_15m[i]
        price = close[i]
        
        # Dynamic position sizing based on volatility
        current_vol = atr / price if price > 0 else 0.01
        vol_adjustment = min(1.5, max(0.7, TARGET_VOL / current_vol)) if current_vol > 0 else 1.0
        position_size = BASE_SIZE * vol_adjustment
        position_size = min(MAX_SIZE, max(MIN_SIZE, position_size))
        
        # Discrete signal levels to reduce churn
        if position_size < 0.23:
            position_size = 0.20
        elif position_size < 0.30:
            position_size = 0.25
        else:
            position_size = 0.35
        
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
        
        # Entry logic: 4h HMA trend + 15m Stoch/RSI + BBW regime
        # Regime filter: not in extreme squeeze or expansion
        regime_ok = BBW_REGIME_MIN <= bbw_val <= BBW_REGIME_MAX
        
        if trend == 1 and regime_ok:  # Bullish trend on 4h
            # Long entry: Stoch oversold + RSI healthy + price above KAMA
            if (stoch_val <= STOCH_LONG_MAX and
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and
                close[i] > kama_15m[i] and
                hma_15m_fast[i] > hma_15m_slow[i]):
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1 and regime_ok:  # Bearish trend on 4h
            # Short entry: Stoch overbought + RSI healthy + price below KAMA
            if (stoch_val >= STOCH_SHORT_MIN and
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and
                close[i] < kama_15m[i] and
                hma_15m_fast[i] < hma_15m_slow[i]):
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