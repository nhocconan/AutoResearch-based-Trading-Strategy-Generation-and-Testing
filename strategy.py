#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian Breakout + Volume + Williams %R + 1w Trend

HYPOTHESIS: Daily Donchian(20) breakouts capture institutional moves that last 
multiple days. Combined with Williams %R extremes for momentum confirmation 
and 1w HMA trend alignment, this strategy filters for high-probability setups.
1d timeframe naturally limits trade frequency to 7-25/year, reducing fee drag.
Works in both bull markets (long breakouts) and bear/range (short rallies to 
channel + mean-reversion at extremes).

TIMEFRAME: 1d primary
HTF: 1w for trend alignment
TARGET: 30-80 total trades over 4 years (7-20/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_williams_1w_v1"
timeframe = "1d"
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
    """Williams %R"""
    n = len(close)
    williams = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            williams[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return williams

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend alignment (bull/bear/range)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate local 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period (20 days = ~1 month)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Williams %R (14-period for momentum)
    williams_r = calculate_williams_r(high, low, close, period=14)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Position tracking
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing for 1d
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(williams_r[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND ALIGNMENT (1w HMA) ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === WILLIAMS %R VALUE ===
        wr_val = williams_r[i]
        
        # === DONCHIAN CHANNEL STATUS ===
        mid_channel = (donch_upper[i] + donch_lower[i]) / 2
        channel_width = donch_upper[i] - donch_lower[i]
        
        # Breakout detection: price closes outside channel
        breakout_up = close[i] > donch_upper[i]
        breakout_down = close[i] < donch_lower[i]
        
        # Near channel edge (for continuation)
        near_upper = close[i] > donch_upper[i] - 0.1 * channel_width
        near_lower = close[i] < donch_lower[i] + 0.1 * channel_width
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Price breaks above upper channel + extreme oversold in bull trend
            if breakout_up:
                # Need volume spike AND bull trend AND Williams %R showing momentum
                if vol_spike and price_above_1w_hma and wr_val < -50:
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Price breaks below lower channel in bear trend OR rally to overbought
            if breakout_down:
                if vol_spike and not price_above_1w_hma and wr_val > -50:
                    desired_signal = -SIZE
                # Also short in bull corrections: rally to overbought + volume
                elif price_above_1w_hma and wr_val > -20 and vol_spike:
                    desired_signal = -SIZE * 0.5  # Half size for counter-trend
        
        # === STOPLOSS CHECK (3 ATR — wider for 1d swings) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price breaks below lower channel OR Williams %R reaches overbought
            if close[i] < donch_lower[i]:
                exit_triggered = True
            if wr_val > -20:  # Overbought
                exit_triggered = True
            # Take profit: Williams %R reaches -5 (very overbought)
            if wr_val > -5:
                desired_signal = SIZE * 0.5  # Reduce position
                exit_triggered = False
        
        if in_position and position_side < 0:
            # Short exit: price breaks above upper channel OR Williams %R reaches oversold
            if close[i] > donch_upper[i]:
                exit_triggered = True
            if wr_val < -80:  # Oversold
                exit_triggered = True
            # Take profit: Williams %R reaches -95 (very oversold)
            if wr_val < -95:
                desired_signal = -SIZE * 0.5
                exit_triggered = False
        
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals