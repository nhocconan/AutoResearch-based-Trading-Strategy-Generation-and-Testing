#!/usr/bin/env python3
"""
EXPERIMENT #028 - MTF HMA Slope + 15m Pullback (4h+15m v3)
==================================================================================================
Hypothesis: #027 failed (Sharpe=0.266) because the 4h trend filter was too restrictive
(price vs HMA creates choppy signals). #021 succeeded (Sharpe=4.629) with pure 15m.

Key changes vs #027:
- 4h trend: Use HMA slope direction (not price vs HMA) - smoother, less whipsaw
- RSI range: 30-70 (wider than #027's 45-55, matching #021's flexibility)
- Remove volume filter (was filtering good entries in #027)
- Remove BBW filter (simplify entry logic)
- Keep: 15m HMA+KAMA agreement, Stochastic timing, Z-score filter
- Stoploss: 2.5*ATR (slightly looser than #027's 2.0*ATR)
- Position size: 0.30 (slightly more conservative than #027's 0.35)

Why this should work:
- 4h HMA slope is smoother than price vs HMA (less false trend changes)
- 15m entries capture optimal timing within 4h trend
- Wider RSI range allows more entries during strong trends
- Simpler logic = fewer filters rejecting good setups

Expected outcome: Sharpe > 4.6 (beat #021), DD < -15%, Trades > 100
"""

import numpy as np
import pandas as pd

name = "mtf_hma_slope_stoch_rsi_zscore_4h_15m_v3"
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
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


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


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    n = len(close)
    if n < k_period:
        return np.zeros(n), np.zeros(n)
    
    k = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        
        if highest_high > lowest_low:
            k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            k[i] = 50
    
    for i in range(k_period - 1 + d_period - 1, n):
        d[i] = np.mean(k[i - d_period + 1:i + 1])
    
    return k, d


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=16)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    stoch_k_15m, stoch_d_15m = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    
    # Resample to 4h for trend filters using proper method
    try:
        prices_indexed = prices.set_index('open_time')
        df_4h = prices_indexed.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calculate 4h indicators
        c_4h = df_4h['close'].values
        hma_4h = calculate_hma(c_4h, period=48)
        
        # Calculate 4h HMA slope (direction) - smoother than price vs HMA
        trend_4h = np.zeros(len(c_4h))
        lookback_slope = 3
        
        for i in range(lookback_slope, len(c_4h)):
            if hma_4h[i] > hma_4h[i - lookback_slope]:
                trend_4h[i] = 1
            elif hma_4h[i] < hma_4h[i - lookback_slope]:
                trend_4h[i] = -1
        
        # Create 4h index for reindexing
        df_4h['trend'] = trend_4h
        
        # Reindex to 15m with ffill
        df_4h_reindexed = df_4h.reindex(prices_indexed.index, method='ffill')
        
        trend_4h_mapped = df_4h_reindexed['trend'].values
        
    except Exception:
        # Fallback: simple downsampling if resample fails
        bars_per_4h = 16  # 15m bars per 4h
        n_4h = (n // bars_per_4h)
        
        c_4h = np.zeros(n_4h)
        for i in range(n_4h):
            start_idx = i * bars_per_4h
            end_idx = start_idx + bars_per_4h
            c_4h[i] = close[end_idx - 1]
        
        hma_4h = calculate_hma(c_4h, period=48)
        
        trend_4h_mapped = np.zeros(n)
        lookback_slope = 3
        
        for i in range(n):
            idx_4h = i // bars_per_4h
            if idx_4h >= lookback_slope and idx_4h < n_4h:
                if hma_4h[idx_4h] > hma_4h[idx_4h - lookback_slope]:
                    trend_4h_mapped[i] = 1
                elif hma_4h[idx_4h] < hma_4h[idx_4h - lookback_slope]:
                    trend_4h_mapped[i] = -1
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries (wider range than #027)
    RSI_LONG_MIN = 30
    RSI_LONG_MAX = 70
    RSI_SHORT_MIN = 30
    RSI_SHORT_MAX = 70
    
    # Stochastic thresholds
    STOCH_LONG_MAX = 80
    STOCH_SHORT_MIN = 20
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.5
    
    # ATR stoploss multiplier (slightly looser than #027)
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 48 * 16, 14 * 2, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    atr_at_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h_mapped[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        stoch_k = stoch_k_15m[i]
        stoch_d = stoch_d_15m[i]
        
        # 15m trend confirmation (HMA > KAMA for long, HMA < KAMA for short)
        hma_kama_bullish = hma_15m[i] > kama_15m[i] if hma_15m[i] > 0 and kama_15m[i] > 0 else False
        hma_kama_bearish = hma_15m[i] < kama_15m[i] if hma_15m[i] > 0 and kama_15m[i] > 0 else False
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            prev_atr = atr_at_entry[i - 1] if atr_at_entry[i - 1] > 0 else atr
            
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
                stoploss_price = prev_entry - ATR_STOP_MULT * prev_atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    atr_at_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * prev_atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    atr_at_entry[i] = prev_atr
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * prev_atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        atr_at_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * prev_atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    atr_at_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * prev_atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    atr_at_entry[i] = prev_atr
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * prev_atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        atr_at_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            atr_at_entry[i] = atr_at_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 15m HMA/KAMA + RSI pullback + Stoch + Z-score
        if trend == 1 and hma_kama_bullish:  # Bullish trend on 4h + 15m confirmation
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                stoch_k < STOCH_LONG_MAX and
                abs(zscore_val) < ZSCORE_MAX):
                signals[i] = BASE_SIZE
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                atr_at_entry[i] = atr
                
        elif trend == -1 and hma_kama_bearish:  # Bearish trend on 4h + 15m confirmation
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                stoch_k > STOCH_SHORT_MIN and
                abs(zscore_val) < ZSCORE_MAX):
                signals[i] = -BASE_SIZE
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                atr_at_entry[i] = atr
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals