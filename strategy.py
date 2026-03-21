#!/usr/bin/env python3
"""
EXPERIMENT #043 - MTF Supertrend+HMA+Zscore+RSI (1h+4h v1)
==================================================================================================
Hypothesis: Current best #034 uses 15m+4h with Sharpe=10.16. 
Testing 1h+4h combination to reduce trade frequency and fees while maintaining edge.
Key changes:
- Timeframe: 1h entries instead of 15m (fewer false signals, less fee drag)
- 4h trend: Supertrend(ATR=10, mult=3) instead of HMA+KAMA (more robust trend filter)
- Entry filter: Z-score(20) for extreme pullback entries
- Position size: 0.30 (more conservative than 0.35)
- Stoploss: 1.5*ATR (tighter than 2.0*ATR)
- Take profit: 2.5R (higher reward ratio)
- Add volume ratio filter for confirmation

Expected: Higher Sharpe due to fewer trades, better fee efficiency, tighter risk control.
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_hma_zscore_rsi_1h_4h_v1"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        
        upper_band = hl2 + multiplier * atr[i]
        lower_band = hl2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = -1
        else:
            if direction[i - 1] == 1:
                if close[i] > supertrend[i - 1]:
                    supertrend[i] = lower_band
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band
                    direction[i] = -1
            else:
                if close[i] < supertrend[i - 1]:
                    supertrend[i] = upper_band
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band
                    direction[i] = 1
    
    return supertrend, direction


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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized distance from mean)"""
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio (current volume vs average)"""
    n = len(volume)
    if n < period:
        return np.ones(n)
    
    vol_ratio = np.ones(n)
    
    for i in range(period - 1, n):
        avg_vol = np.mean(volume[i - period + 1:i + 1])
        if avg_vol > 0:
            vol_ratio[i] = volume[i] / avg_vol
        else:
            vol_ratio[i] = 1.0
    
    return vol_ratio


def resample_to_4h(prices):
    """Resample 1h data to 4h using proper open_time index"""
    if 'open_time' not in prices.columns:
        return None
    
    prices_indexed = prices.set_index('open_time')
    
    df_4h = prices_indexed.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_4h


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    zscore_1h = calculate_zscore(close, period=20)
    vol_ratio_1h = calculate_volume_ratio(volume, period=20)
    _, _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Resample to 4h for trend filters using proper method
    df_4h = resample_to_4h(prices)
    
    # Initialize 4h indicators mapped to 1h
    trend_4h = np.zeros(n)
    supertrend_4h_mapped = np.zeros(n)
    direction_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    c_4h_mapped = np.zeros(n)
    
    if df_4h is not None and len(df_4h) > 0:
        c_4h = df_4h["close"].values
        h_4h = df_4h["high"].values
        l_4h = df_4h["low"].values
        
        # Calculate 4h Supertrend
        supertrend_4h, direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
        
        # Map 4h indicators back to 1h timeframe using open_time
        prices_indexed = prices.set_index('open_time')
        
        # Create 4h series with proper timestamps
        supertrend_4h_series = pd.Series(supertrend_4h, index=df_4h.index)
        direction_4h_series = pd.Series(direction_4h, index=df_4h.index)
        bbw_4h_series = pd.Series(bbw_4h, index=df_4h.index)
        c_4h_series = pd.Series(c_4h, index=df_4h.index)
        
        # Reindex to 1h with forward fill
        supertrend_4h_mapped_series = supertrend_4h_series.reindex(prices_indexed.index).ffill()
        direction_4h_mapped_series = direction_4h_series.reindex(prices_indexed.index).ffill()
        bbw_4h_mapped_series = bbw_4h_series.reindex(prices_indexed.index).ffill()
        c_4h_mapped_series = c_4h_series.reindex(prices_indexed.index).ffill()
        
        # Fill remaining NaN values
        supertrend_4h_mapped = supertrend_4h_mapped_series.fillna(0).values
        direction_4h_mapped = direction_4h_mapped_series.fillna(0).values
        bbw_4h_mapped = bbw_4h_mapped_series.fillna(0).values
        c_4h_mapped = c_4h_mapped_series.fillna(0).values
        
        # Calculate 4h trend from Supertrend direction
        for i in range(n):
            trend_4h[i] = direction_4h_mapped[i]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # Z-score thresholds for extreme entries
    ZSCORE_LONG_MAX = -0.5  # Entry on pullback (below mean)
    ZSCORE_SHORT_MIN = 0.5  # Entry on rally (above mean)
    
    # BBW minimum for regime filter (4h)
    BBW_MIN = 0.01
    
    # ATR stoploss multiplier (tighter than before)
    ATR_STOP_MULT = 1.5
    ATR_TP_MULT = 2.5  # Higher reward ratio
    
    # Volume ratio minimum
    VOL_RATIO_MIN = 0.8
    
    first_valid = max(200, 14 * 2, 20, 21 + int(np.sqrt(21)))
    
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
        atr = atr_1h[i]
        price = close[i]
        bbw_4h_val = bbw_4h_mapped[i]
        zscore_val = zscore_1h[i]
        vol_ratio = vol_ratio_1h[i]
        
        # 4h trend must exist
        if trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # BBW filter - avoid choppy markets (4h)
        if bbw_4h_val < BBW_MIN:
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
            
            # Stoploss check (1.5*ATR)
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
                
                # Take profit check (2.5R) - reduce to half
                tp_price = prev_entry + ATR_TP_MULT * atr
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
                
                # Take profit check (2.5R) - reduce to half
                tp_price = prev_entry - ATR_TP_MULT * atr
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
        
        # Volume filter
        if vol_ratio < VOL_RATIO_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Entry logic: 4h Supertrend + 1h HMA + Z-score + RSI
        if trend == 1:  # Bullish trend on 4h
            # 1h trend confirmation (HMA) + pullback entry (Z-score + RSI)
            if (close[i] > hma_1h[i] and
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and
                zscore_val <= ZSCORE_LONG_MAX):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend == -1:  # Bearish trend on 4h
            # 1h trend confirmation (HMA) + pullback entry (Z-score + RSI)
            if (close[i] < hma_1h[i] and
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and
                zscore_val >= ZSCORE_SHORT_MIN):
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
    
    return signals