#!/usr/bin/env python3
"""
Experiment #006: 4h Donchian Breakout + 1w SMA Trend + Volume + Choppiness

HYPOTHESIS: Donchian(20) breakout captures institutional momentum moves.
Combined with 1w SMA trend filter, volume confirmation, and choppiness regime,
this captures trending moves in both bull and bear markets while avoiding
whipsaws in choppy conditions.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Donchian breakout is symmetric (works for long and short)
- 1w SMA provides longer-term trend context
- Bear markets: shorts when price < 1w SMA, breakout below Donchian low
- Bull markets: longs when price > 1w SMA, breakout above Donchian high
- Choppiness filter prevents trading in sideways markets

TARGET: 75-150 total trades over 4 years (proven pattern).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382)

KEY DESIGN:
1. Donchian(20) breakout as entry signal
2. 1w SMA(20) for trend direction filter
3. Volume confirmation (>1.5x 20-avg)
4. Choppiness < 50 (only trending markets)
5. ATR-based stoploss (2.5*ATR) and profit target (2R)
6. Signal: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1w_sma_vol_chop_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging (no trades), CHOP < 50 = trending (allow trades)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend SMA (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w SMA(20) for trend
    sma_1w_raw = pd.Series(df_1w['close'].values).rolling(window=20, min_periods=20).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    # Warmup - need at least 20 bars for Donchian + ATR
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(sma_1w_aligned[i]):
            continue
        
        # === REGIME CHECK (Choppiness) ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # Only trade in trending markets
        
        if not is_trending:
            # Exit if in position, no new entries
            if in_position:
                signals[i] = 0.0
                # Process exits
                stoploss_hit = False
                if position_side > 0 and low[i] < stop_price:
                    stoploss_hit = True
                if position_side < 0 and high[i] > stop_price:
                    stoploss_hit = True
                
                if stoploss_hit:
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    stop_price = 0.0
            continue
        
        # === TREND DIRECTION (1w SMA) ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Use previous close for confirmation (avoid look-ahead)
        if i < 1:
            continue
        
        prev_close = close[i - 1]
        prev_atr = atr_14[i - 1]
        
        # Long: breakout above Donchian high with bullish trend
        long_signal = False
        if price_above_1w_sma and prev_close > donchian_high[i - 1]:
            long_signal = True
        
        # Short: breakdown below Donchian low with bearish trend
        short_signal = False
        if not price_above_1w_sma and prev_close < donchian_low[i - 1]:
            short_signal = True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if long_signal and vol_spike:
            desired_signal = SIZE
        elif short_signal and vol_spike:
            desired_signal = -SIZE
        
        # === STOPLOSS AND TAKE PROFIT ===
        stoploss_hit = False
        tp_hit = False
        
        if in_position:
            if position_side > 0:
                # Update highest and trailing stop
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                
                if low[i] < stop_price:
                    stoploss_hit = True
                
                # TP at 2R (4*ATR from entry)
                tp_price = entry_price + 4.0 * entry_atr
                if high[i] >= tp_price:
                    tp_hit = True
                    
            elif position_side < 0:
                # Update lowest and trailing stop
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                
                if high[i] > stop_price:
                    stoploss_hit = True
                
                # TP at 2R
                tp_price = entry_price - 4.0 * entry_atr
                if low[i] <= tp_price:
                    tp_hit = True
        
        # === EXECUTE TRADES ===
        if in_position:
            if stoploss_hit or tp_hit:
                # Exit position
                signals[i] = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                stop_price = 0.0
            else:
                # Maintain position
                signals[i] = SIZE if position_side > 0 else -SIZE
                
        elif desired_signal != 0.0:
            # New entry
            signals[i] = desired_signal
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
    
    return signals