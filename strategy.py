#!/usr/bin/env python3
"""
Experiment #021: 4h Vortex Crossover + Volume + 12h Trend

HYPOTHESIS: The Vortex Indicator (VI) identifies trend direction changes 
more reliably than RSI or MACD because it's derived from actual price 
positioning (high/low vs prior bars). VI crossovers are RARE events that 
mark institutional momentum shifts. Combined with volume confirmation 
and 12h HMA trend alignment, this captures high-probability setups in 
both bull (long VI+ crossover) and bear (short VI- crossover) markets.

TIMEFRAME: 4h primary
HTF: 12h for trend alignment via HMA
TARGET: 75-200 total trades over 4 years (19-50/year)
REASONING: VI crossovers happen ~1-2x/month per symbol = 48-96 trades/4y
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vortex_vol_12h_trend_v1"
timeframe = "4h"
leverage = 1.0

def calculate_vortex(close, high, low, period=14):
    """Vortex Indicator - identifies trend reversal points"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    vm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if np.isnan(close[i-1]) or np.isnan(high[i-1]) or np.isnan(low[i-1]):
            continue
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
        vm[i] = abs(high[i] - low[i-1]) - abs(low[i] - high[i-1])
    
    vi_plus = pd.Series(vm).ewm(span=period, min_periods=period, adjust=False).mean().values
    vi_minus = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return vi_plus, vi_minus, tr

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA for trend alignment
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    vi_plus, vi_minus, _ = calculate_vortex(close, high, low, period=14)
    
    # VI signal line (9-period EMA of VI)
    vi_signal_raw = pd.Series(vi_plus - vi_minus).ewm(span=9, min_periods=9, adjust=False).mean().values
    vi_signal_aligned = align_htf_to_ltf(prices, df_12h, vi_signal_raw)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Need 100 bars for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vi_plus[i]) or np.isnan(vi_signal_aligned[i]):
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
        
        # Current values
        vi_current = vi_plus[i] - vi_minus[i]
        vi_signal_current = vi_signal_aligned[i]
        
        # Previous VI difference (from previous bar)
        vi_prev = (vi_plus[i-1] - vi_minus[i-1]) if i > 0 else 0.0
        vi_signal_prev = vi_signal_aligned[i-1] if i > 0 else 0.0
        
        # Crossover detection
        bullish_cross = (vi_current > vi_signal_current) and (vi_prev <= vi_signal_prev)
        bearish_cross = (vi_current < vi_signal_current) and (vi_prev >= vi_signal_prev)
        
        # Volume confirmation
        vol_confirm = vol_ratio[i] > 1.5
        
        # 12h trend filter
        price_above_12h_hma = close[i] > hma_12h_aligned[i]
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # VI bullish crossover + volume spike + price above 12h HMA
            if bullish_cross and vol_confirm and price_above_12h_hma:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # VI bearish crossover + volume spike + price below 12h HMA
            if bearish_cross and vol_confirm and not price_above_12h_hma:
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
        
        # === OPPOSITE SIGNAL EXIT ===
        if in_position and position_side > 0 and bearish_cross:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and bullish_cross:
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
        
        signals[i] = desired_signal
    
    return signals