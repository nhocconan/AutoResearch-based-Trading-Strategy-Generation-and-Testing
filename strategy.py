#!/usr/bin/env python3
"""
Experiment #022: 12h Bollinger Band Mean Reversion + Volume + 1d HMA

HYPOTHESIS: On 12h timeframe, Bollinger Band touches at extremes combined with 
volume confirmation mark high-probability mean reversion setups. The 1d HMA 
provides trend context to filter directional bias. This works in BOTH bull 
markets (long lower BB bounces) and bear markets (short upper BB rallies) because 
it only enters in the trend direction determined by 1d HMA.

TIMEFRAME: 12h primary
HTF: 1d for trend filter
TARGET: 75-200 total trades over 4 years (19-50/year) - conservative for 12h
ENTRY: Lower BB touch + volume spike + 1d HMA above price (bullish alignment)
       Upper BB touch + volume spike + 1d HMA below price (bearish alignment)
EXIT: ATR trailing stop + opposite RSI extreme or middle BB touch
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_bb_volume_1d_hma_v1"
timeframe = "12h"
leverage = 1.0

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

def calculate_bollinger_bands(close, period=20, num_std=2.5):
    """Bollinger Bands - returns upper, middle, lower"""
    n = len(close)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + num_std * std
    lower = middle - num_std * std
    
    return upper, middle, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === Calculate local indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20, num_std=2.5)
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    bars_since_entry = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        bars_since_entry = i - entry_bar
        
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Current indicator values
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # Volume confirmation required
        vol_spike = vol_ratio_val > 1.2
        
        # === ENTRY CONDITIONS ===
        # Lower BB touch + RSI not oversold + bullish 1d trend + volume
        touch_lower_bb = low[i] <= bb_lower[i]
        long_rsi_ok = 30 < rsi_val < 55  # Not extreme oversold
        
        # Upper BB touch + RSI not overbought + bearish 1d trend + volume
        touch_upper_bb = high[i] >= bb_upper[i]
        short_rsi_ok = 45 < rsi_val < 70  # Not extreme overbought
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Trailing stop for longs
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            
            if low[i] < stop_price:
                stoploss_triggered = True
            
            # RSI exit for longs (when RSI reaches extreme)
            if rsi_val > 70:
                exit_triggered = True
            
            # Take profit at middle BB
            if close[i] >= bb_middle[i] and bars_since_entry > 8:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Trailing stop for shorts
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            
            if high[i] > stop_price:
                stoploss_triggered = True
            
            # RSI exit for shorts (when RSI reaches extreme)
            if rsi_val < 30:
                exit_triggered = True
            
            # Take profit at middle BB
            if close[i] <= bb_middle[i] and bars_since_entry > 8:
                exit_triggered = True
        
        # === POSITION MANAGEMENT ===
        desired_signal = 0.0
        
        if stoploss_triggered or exit_triggered:
            desired_signal = 0.0
        elif not in_position:
            # === NEW LONG ENTRY ===
            # Price touches lower BB + RSI in middle range + above 1d HMA + volume
            if touch_lower_bb and long_rsi_ok and price_above_1d_hma and vol_spike:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Price touches upper BB + RSI in middle range + below 1d HMA + volume
            if touch_upper_bb and short_rsi_ok and not price_above_1d_hma and vol_spike:
                desired_signal = -SIZE
        
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