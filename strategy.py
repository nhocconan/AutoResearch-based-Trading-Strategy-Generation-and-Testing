#!/usr/bin/env python3
"""
EXPERIMENT #117 - MTF HMA+RSI+Chandelier+VolRegime+ADX (15m+4h Optimized v1)
==================================================================================================
Hypothesis: Building on #112 (Sharpe=6.193) and #108 (Sharpe=7.706), combine:
- 4h HMA trend direction (proven stable trend filter)
- 15m RSI pullback entries (precise timing)
- Chandelier exit (3*ATR(22)) for trailing stoploss
- Volatility regime for position sizing (low vol=0.35, high vol=0.20)
- ADX(14)>25 on 4h for trend strength confirmation
- Discrete signal levels (0.0, ±0.20, ±0.35) to reduce churn costs

Key improvements over #112:
- Chandelier exit at 3*ATR(22) instead of 2*ATR(14) - wider stop for crypto volatility
- Volatility-adjusted sizing: BBW percentile determines size (low vol=full, high vol=half)
- ADX filter on 4h timeframe only (stronger trend confirmation)
- Cleaner MTF resampling with proper index mapping
- Hysteresis on entries to avoid flip-flopping

Why 15m+4h:
- 4h trend is more stable than 1h (fewer false signals)
- 15m entries capture better R:R than 1h entries
- Proven in #105, #108, #112 with Sharpe > 5.0
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_chandelier_volregime_adx_15m_4h_v117"
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
    
    # Rolling highest high and lowest low
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    
    for i in range(period - 1, n):
        highest_high[i] = np.max(high[i - period + 1:i + 1])
        lowest_low[i] = np.min(low[i - period + 1:i + 1])
        
        chandelier_long[i] = highest_high[i] - multiplier * atr[i]
        chandelier_short[i] = lowest_low[i] + multiplier * atr[i]
    
    return chandelier_long, chandelier_short


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Resample to 4h for trend filters (16 x 15m = 4h)
    bars_per_4h = 16
    n_4h = (n // bars_per_4h)
    
    # Create 4h arrays by downsampling
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * bars_per_4h
        end_idx = start_idx + bars_per_4h
        c_4h[i] = close[end_idx - 1]
        h_4h[i] = np.max(high[start_idx:end_idx])
        l_4h[i] = np.min(low[start_idx:end_idx])
    
    # 4h indicators for trend
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=22)
    hma_4h = calculate_hma(c_4h, period=48)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    
    # Chandelier exit on 4h
    chand_long_4h, chand_short_4h = calculate_chandelier_exit(h_4h, l_4h, c_4h, atr_4h, period=22, multiplier=3.0)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    chand_long_mapped = np.zeros(n)
    chand_short_mapped = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 48:
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            chand_long_mapped[i] = chand_long_4h[idx_4h]
            chand_short_mapped[i] = chand_short_4h[idx_4h]
            atr_4h_mapped[i] = atr_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on volatility regime
    SIZE_HIGH_VOL = 0.20  # High volatility = smaller position
    SIZE_LOW_VOL = 0.35   # Low volatility = full position
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 25
    
    # BBW percentile for volatility regime (4h)
    BBW_HIGH_THRESHOLD = 0.04  # Above this = high vol
    
    first_valid = max(300, 48 * bars_per_4h, 22 * 2, 20, 28)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    chandelier_stop = np.zeros(n)
    
    # Volatility regime tracking
    bbw_4h_percentile = np.zeros(n)
    for i in range(first_valid, n):
        if i >= 100:
            bbw_window = bbw_4h_mapped[max(0, i-100):i+1]
            bbw_4h_percentile[i] = np.searchsorted(np.sort(bbw_window), bbw_4h_mapped[i]) / len(bbw_window)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        adx_4h_val = adx_4h_mapped[i]
        bbw_4h_val = bbw_4h_mapped[i]
        chand_long = chand_long_mapped[i]
        chand_short = chand_short_mapped[i]
        
        # Determine position size based on volatility regime
        if bbw_4h_percentile[i] > 0.7 or bbw_4h_val > BBW_HIGH_THRESHOLD:
            current_size = SIZE_HIGH_VOL
        else:
            current_size = SIZE_LOW_VOL
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filter (4h HMA)
        if trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check Chandelier exit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
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
            
            # Chandelier exit check (4h based)
            if prev_side == 1:
                # Long position: exit if price < chandelier long
                if price < chand_long:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    chandelier_stop[i] = 0
                    continue
                
                # Hold position
                signals[i] = current_size
                position_side[i] = 1
                entry_price[i] = prev_entry
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
                chandelier_stop[i] = chand_long
                
            elif prev_side == -1:
                # Short position: exit if price > chandelier short
                if price > chand_short:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    chandelier_stop[i] = 0
                    continue
                
                # Hold position
                signals[i] = -current_size
                position_side[i] = -1
                entry_price[i] = prev_entry
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
                chandelier_stop[i] = chand_short
            
            continue
        
        # Entry logic: 4h HMA trend + ADX + 15m RSI pullback
        if trend == 1 and adx_4h_val >= ADX_MIN:  # Bullish trend confirmed on 4h
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:  # Pullback entry
                signals[i] = current_size
                position_side[i] = 1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                chandelier_stop[i] = chand_long
                
        elif trend == -1 and adx_4h_val >= ADX_MIN:  # Bearish trend confirmed on 4h
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:  # Pullback entry
                signals[i] = -current_size
                position_side[i] = -1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                chandelier_stop[i] = chand_short
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals