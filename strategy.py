#!/usr/bin/env python3
"""
Experiment #022: 4h Bollinger Squeeze Breakout + 1d Trend

HYPOTHESIS: Bollinger Band squeeze (low volatility) followed by volatility 
expansion captures institutional moves. BB width percentile < 10 = squeeze, 
then BB breakout = entry. Combined with 1d HMA trend direction and volume 
confirmation. BB squeeze breaks 5-10x per year vs Donchian's 2-3x, giving 
proper trade frequency. ATR stoploss manages risk. Works in both bull 
(breakout up) and bear (breakout down with 1d trend filter).

TIMEFRAME: 4h primary
HTF: 1d for trend alignment
TARGET: 100-250 total trades over 4 years (25-60/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_trend_1d_v1"
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

def calculate_bollinger_bands(close, period=20, num_std=2):
    """Bollinger Bands - returns (upper, middle, lower, width)"""
    n = len(close)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = middle + num_std * std
    lower = middle - num_std * std
    
    # BB width for squeeze detection
    width = upper - lower
    
    return upper, middle, lower, width

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

def calculate_bb_width_percentile(width, period=100):
    """Percentile rank of current BB width vs recent range"""
    n = len(width)
    percentile = np.full(n, 50.0, dtype=np.float64)  # default middle
    
    for i in range(period, n):
        if not np.isnan(width[i]):
            window = width[i - period + 1:i + 1]
            if not np.any(np.isnan(window)):
                rank = np.sum(window < width[i]) / len(window) * 100
                percentile[i] = rank
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, num_std=2)
    
    # BB width percentile (squeeze detection)
    bb_width_pct = calculate_bb_width_percentile(bb_width, period=100)
    
    # Volume metrics
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for momentum
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
    
    # Squeeze state tracking
    squeeze_active = False
    squeeze_bar = 0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_width_pct[i]):
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
        
        # === 1d TREND ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        trend_bullish = price_above_1d_hma
        
        # === VOLUME ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === RSI ===
        rsi_val = rsi[i]
        
        # === SQUEEZE DETECTION ===
        # BB width in bottom 15% = squeeze
        in_squeeze = bb_width_pct[i] < 15.0
        
        # === BREAKOUT DETECTION ===
        # Price breaks above BB upper
        breakout_up = close[i] > bb_upper[i] and close[i-1] <= bb_upper[i-1] if i > warmup else False
        # Price breaks below BB lower
        breakout_down = close[i] < bb_lower[i] and close[i-1] >= bb_lower[i-1] if i > warmup else False
        
        # Price already outside bands
        price_above_bb_upper = close[i] > bb_upper[i]
        price_below_bb_lower = close[i] < bb_lower[i]
        
        # === ATR STOPLOSS LEVELS ===
        stop_atr_mult = 2.5
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW ENTRY: Squeeze + Breakout + Volume + Trend ===
            
            # LONG: breakout up with squeeze, volume, bullish trend
            if (breakout_up or price_above_bb_upper) and in_squeeze:
                if vol_spike and trend_bullish:
                    desired_signal = SIZE
                    squeeze_active = True
                    squeeze_bar = i
            
            # SHORT: breakout down with squeeze, volume, bearish trend
            if (breakout_down or price_below_bb_lower) and in_squeeze:
                if vol_spike and not trend_bullish:
                    desired_signal = -SIZE
                    squeeze_active = True
                    squeeze_bar = i
            
            # Also allow entries without prior squeeze if trend is very strong
            # LONG: breakout up, very strong volume, strong trend
            if not in_squeeze and trend_bullish:
                if vol_spike and vol_ratio[i] > 2.0 and breakout_up:
                    desired_signal = SIZE
            
            # SHORT: breakout down, very strong volume, strong downtrend
            if not in_squeeze and not trend_bullish:
                if vol_spike and vol_ratio[i] > 2.0 and breakout_down:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - stop_atr_mult * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + stop_atr_mult * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT: 3R or RSI extreme ===
        tp_triggered = False
        if in_position and position_side > 0:
            profit = (close[i] - entry_price) / entry_price
            if profit > 0.06:  # 6% profit (~3R with 2% ATR risk)
                tp_triggered = True
            if rsi_val > 75:  # RSI overbought
                tp_triggered = True
        
        if in_position and position_side < 0:
            profit = (entry_price - close[i]) / entry_price
            if profit > 0.06:
                tp_triggered = True
            if rsi_val < 25:  # RSI oversold
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === EXIT: Opposite band touch ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            if price_below_bb_lower:  # Fell back through lower band
                exit_triggered = True
        
        if in_position and position_side < 0:
            if price_above_bb_upper:  # Broke back through upper band
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                squeeze_active = False
                if position_side > 0:
                    stop_price = entry_price - stop_atr_mult * entry_atr
                else:
                    stop_price = entry_price + stop_atr_mult * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                squeeze_active = False
        
        signals[i] = desired_signal
    
    return signals