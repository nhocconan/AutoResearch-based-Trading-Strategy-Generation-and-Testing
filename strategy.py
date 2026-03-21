#!/usr/bin/env python3
"""
EXPERIMENT #002 - MTF KAMA+RSI+BBW+Zscore (1h+4h Adaptive Trend v1)
==================================================================================================
Hypothesis: Current best uses 4H HMA trend + 1H RSI pullback. This experiment tests:
- Timeframe: 1h entries + 4h trend (proven MTF combination)
- Trend: KAMA(10) instead of HMA - adaptive to market efficiency, less whipsaw in chop
- Entry: RSI(14) pullback to 40-60 zone instead of extremes - catches trend continuations
- Filter: Bollinger Band Width percentile + Z-score(20) - volatility regime + overextension
- Position size: 0.30 discrete levels with ATR stoploss at 2.5*ATR

Why this might beat #040:
- KAMA adapts to market noise better than HMA (Kaufman's Adaptive Moving Average)
- RSI pullback to 40-60 catches trend continuations better than extreme reversals
- BBW percentile filters low-volatility chop regimes
- Z-score prevents entering at overextended prices
- 1h timeframe balances signal frequency vs transaction costs
"""

import numpy as np
import pandas as pd

name = "mtf_kama_rsi_bbw_zscore_1h_v1"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA)
    KAMA adapts to market noise by adjusting smoothing constant based on Efficiency Ratio
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    er = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    for i in range(er_period - 1, n):
        if i >= er_period:
            change = abs(close[i] - close[i - er_period])
            noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if noise > 0:
                er[i] = change / noise
            else:
                er[i] = 0
    
    # Calculate smoothing constant (SC)
    # SC = ER * (fast_SC - slow_SC) + slow_SC
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with SMA
    kama[er_period - 1] = np.mean(close[:er_period])
    
    for i in range(er_period, n):
        sc = er[i] * (fast_sc - slow_sc) + slow_sc
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    rsi = np.zeros(n)
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    for i in range(1, n):
        diff = close[i] - close[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = abs(diff)
    
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands (upper, middle, lower, bandwidth)"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    middle = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bandwidth[i] = 0
    
    return upper, middle, lower, bandwidth


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


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema[i] = ema[i - 1] + (2.0 / (period + 1)) * (close[i] - ema[i - 1])
    
    return ema


def calculate_supertrend(high, low, close, period=10, mult=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    # Initialize
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period - 1, n):
        upper_band[i] = (high[i] + low[i]) / 2 + mult * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - mult * atr[i]
    
    # First valid supertrend
    supertrend[period - 1] = upper_band[period - 1]
    direction[period - 1] = 1
    
    for i in range(period, n):
        if close[i] > supertrend[i - 1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        
        # Check for trend flip
        if direction[i] == 1 and close[i] < supertrend[i - 1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        elif direction[i] == -1 and close[i] > supertrend[i - 1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
    
    return supertrend, direction


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    signals = np.zeros(n)
    
    # Check if open_time column exists for proper MTF resampling
    if 'open_time' in prices.columns:
        # PROPER MTF: Use actual timestamps for resampling
        prices_indexed = prices.set_index('open_time')
        
        # Resample to 4h for trend filter
        df_4h = prices_indexed.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(df_4h) < 60:
            # Not enough 4h data
            return signals
        
        # Calculate 4h indicators
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h KAMA for adaptive trend
        kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
        
        # 4h Supertrend for trend direction
        _, supertrend_dir_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, mult=3.0)
        
        # 4h BBW for volatility regime
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        # Calculate BBW percentile (rolling 50 periods)
        bbw_percentile_4h = np.zeros(len(bbw_4h))
        for i in range(50, len(bbw_4h)):
            window = bbw_4h[i - 50:i + 1]
            bbw_percentile_4h[i] = np.sum(bbw_4h[i - 50:i] <= bbw_4h[i]) / 50.0
        
        # Map 4h trend back to 1h using ffill (proper alignment)
        kama_4h_series = pd.Series(kama_4h, index=df_4h.index)
        supertrend_dir_4h_series = pd.Series(supertrend_dir_4h, index=df_4h.index)
        bbw_pct_4h_series = pd.Series(bbw_percentile_4h, index=df_4h.index)
        
        # Reindex to match 1h timestamps with forward fill
        kama_4h_aligned = kama_4h_series.reindex(prices_indexed.index, method='ffill').fillna(0).values
        supertrend_dir_4h_aligned = supertrend_dir_4h_series.reindex(prices_indexed.index, method='ffill').fillna(0).values
        bbw_pct_4h_aligned = bbw_pct_4h_series.reindex(prices_indexed.index, method='ffill').fillna(0).values
        
    else:
        # Fallback: simple downsampling if no open_time
        bars_per_4h = 4
        n_4h = n // bars_per_4h
        
        if n_4h < 60:
            return signals
        
        close_4h = np.array([close[(i + 1) * bars_per_4h - 1] for i in range(n_4h)])
        high_4h = np.array([np.max(high[i * bars_per_4h:(i + 1) * bars_per_4h]) for i in range(n_4h)])
        low_4h = np.array([np.min(low[i * bars_per_4h:(i + 1) * bars_per_4h]) for i in range(n_4h)])
        
        kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
        _, supertrend_dir_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, mult=3.0)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        kama_4h_aligned = np.zeros(n)
        supertrend_dir_4h_aligned = np.zeros(n)
        bbw_pct_4h_aligned = np.zeros(n)
        
        for i in range(n):
            idx_4h = min(i // bars_per_4h, n_4h - 1)
            if idx_4h >= 50:
                kama_4h_aligned[i] = kama_4h[idx_4h]
                supertrend_dir_4h_aligned[i] = supertrend_dir_4h[idx_4h]
                bbw_pct_4h_aligned[i] = 0.5  # Default neutral
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    _, bb_middle_1h, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI pullback zones
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    
    # Z-score threshold for overextension filter
    ZSCORE_MAX = 2.0
    
    # BBW percentile threshold (avoid low volatility chop)
    BBW_PCT_MIN = 0.30  # Must be above 30th percentile
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 55 * 4, 30 + 10, 50)
    
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
        
        # 4h trend filters
        kama_4h = kama_4h_aligned[i]
        supertrend_dir_4h = supertrend_dir_4h_aligned[i]
        bbw_pct_4h = bbw_pct_4h_aligned[i]
        
        # 1h entry filters
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # 1h KAMA for local trend confirmation
        kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
        
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h KAMA/Supertrend trend + 1h RSI pullback + BBW + Z-score
        # Long entry
        if supertrend_dir_4h == 1 and close[i] > kama_4h and bbw_pct_4h > BBW_PCT_MIN:
            # RSI pullback to 40-55 zone (trend continuation)
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                # Z-score filter (not overextended)
                if abs(zscore_val) < ZSCORE_MAX:
                    # 1h KAMA confirmation
                    if close[i] > kama_1h[i]:
                        signals[i] = SIZE_FULL
                        position_side[i] = 1
                        entry_price[i] = price
                        tp_triggered[i] = 0
                        highest_since_entry[i] = price
                        lowest_since_entry[i] = price
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        # Short entry
        elif supertrend_dir_4h == -1 and close[i] < kama_4h and bbw_pct_4h > BBW_PCT_MIN:
            # RSI pullback to 45-60 zone (trend continuation)
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                # Z-score filter (not overextended)
                if abs(zscore_val) < ZSCORE_MAX:
                    # 1h KAMA confirmation
                    if close[i] < kama_1h[i]:
                        signals[i] = -SIZE_FULL
                        position_side[i] = -1
                        entry_price[i] = price
                        tp_triggered[i] = 0
                        highest_since_entry[i] = price
                        lowest_since_entry[i] = price
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals