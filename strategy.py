#!/usr/bin/env python3
"""
Experiment #021: 4h Bollinger Mean Reversion + Volume + 1d Trend

HYPOTHESIS: Markets oscillate between trends and ranges. On 4h, Bollinger Bands
identify mean reversion opportunities. Combined with volume confirmation and
1d HMA trend alignment, this captures reversals in the direction of the larger trend.
Works in bull markets (buying dips to lower band) and bear markets (selling rallies
to upper band).

TIMEFRAME: 4h primary, 1d for trend filter
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_boll_mean_reversion_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    mask = ~(np.isnan(wma_half) | np.isnan(wma_full))
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, num_std=2):
    """Bollinger Bands - returns middle, upper, lower"""
    n = len(close)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    donch_upper = np.full(n, np.nan, dtype=np.float64)
    donch_lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        donch_upper[i] = np.max(high[i - period + 1:i + 1])
        donch_lower[i] = np.min(low[i - period + 1:i + 1])
    
    return donch_upper, donch_lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger(close, period=20, num_std=2)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # 1d HMA trend direction
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        rsi_val = rsi[i]
        vol_spike = vol_ratio[i] > 1.5
        
        # Bollinger Band position
        price_at_lower = close[i] <= bb_lower[i]
        price_at_upper = close[i] >= bb_upper[i]
        
        # Previous bar check for exact touch
        price_was_above_lower = close[i-1] > bb_lower[i-1] if i > warmup else True
        price_was_below_upper = close[i-1] < bb_upper[i-1] if i > warmup else True
        
        desired_signal = 0.0
        
        # === NEW ENTRY CONDITIONS ===
        if not in_position:
            # LONG: price touches lower band + RSI not extremely oversold + bullish 1d + volume
            # Relaxed: RSI < 55 (not deeply oversold, more entries)
            if price_at_lower and rsi_val < 55 and price_above_1d_hma and vol_spike:
                desired_signal = SIZE
            
            # SHORT: price touches upper band + RSI not extremely overbought + bearish 1d + volume
            if price_at_upper and rsi_val > 45 and not price_above_1d_hma and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price crosses above upper band OR RSI overbought
            if close[i] > bb_upper[i] and price_was_below_upper:
                exit_triggered = True
            if rsi_val > 70:
                exit_triggered = True
            # Also exit if 1d trend flips bearish
            if not price_above_1d_hma:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price crosses below lower band OR RSI oversold
            if close[i] < bb_lower[i] and price_was_above_lower:
                exit_triggered = True
            if rsi_val < 30:
                exit_triggered = True
            # Also exit if 1d trend flips bullish
            if price_above_1d_hma:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction change
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals