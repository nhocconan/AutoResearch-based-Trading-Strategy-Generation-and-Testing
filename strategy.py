#!/usr/bin/env python3
"""
Experiment #020: 4h Williams %R + Donchian + 12h Trend Confirmation

HYPOTHESIS: Williams %R is a momentum oscillator that catches reversals at
channel extremes (similar to CRSI which scored 1.46 on SOL). Combined with
Donchian for structure and 12h HMA for trend bias, this captures:
- Bull markets: Long when %R oversold (<-80) + price above 12h HMA + Donchian touch
- Bear markets: Short when %R overbought (> -20) + price below 12h HMA
- Range: Mean-revert at %R extremes

Williams %R has different signal characteristics than RSI/CRSI, potentially
offering better timing on 4h timeframe. Tighter entry = fewer but higher quality trades.

TIMEFRAME: 4h primary
HTF: 12h for trend bias (HMA21)
TARGET: 75-200 total trades over 4 years (19-50/year)
SIZE: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_williams_donchian_12h_v1"
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
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 1e-10:
            willr[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            willr[i] = -50.0  # Neutral when no range
    
    return willr

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average"""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return volume / np.where(vol_ma > 0, vol_ma, 1)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate local 4h indicators
    willr = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
    entry_bar = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(willr[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WILLIAMS %R VALUES ===
        willr_val = willr[i]
        
        # %R oversold: below -80 (potential long entry)
        # %R overbought: above -20 (potential short entry)
        willr_oversold = willr_val < -80
        willr_overbought = willr_val > -20
        
        # %R extreme for exit
        willr_extreme_long_exit = willr_val > -20  # Getting overbought
        willr_extreme_short_exit = willr_val < -80  # Getting oversold
        
        # === TREND BIAS (12h HMA) ===
        price_above_12h_hma = close[i] > hma_12h_aligned[i]
        
        # === DONCHIAN TOUCH (near channel extremes) ===
        mid_channel = (donch_upper[i] + donch_lower[i]) / 2
        channel_width = donch_upper[i] - donch_lower[i]
        
        # Price near upper band (within 20% of channel width)
        near_upper = close[i] >= donch_upper[i] - 0.2 * channel_width
        # Price near lower band
        near_lower = close[i] <= donch_lower[i] + 0.2 * channel_width
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Williams %R oversold + near lower Donchian + price above 12h HMA + volume
            if willr_oversold and near_lower and price_above_12h_hma and vol_spike:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Williams %R overbought + near upper Donchian + price below 12h HMA + volume
            if willr_overbought and near_upper and not price_above_12h_hma and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
            # Long exit: %R reaches overbought OR price breaks above upper Donchian
            if willr_extreme_long_exit:
                exit_triggered = True
            if close[i] > donch_upper[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: %R reaches oversold OR price breaks below lower Donchian
            if willr_extreme_short_exit:
                exit_triggered = True
            if close[i] < donch_lower[i]:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals