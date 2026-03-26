#!/usr/bin/env python3
"""
Experiment #027: 12h Bollinger Bounce + 1d Trend + Volume Spike

HYPOTHESIS: In the current bear/range market (2025 test), mean reversion to 
Bollinger Bands outperforms trend-following. Price tends to bounce from BB 
extremes rather than trending. On 12h, BB bounces mark high-probability 
reversal points. Combined with 1d HMA trend alignment (to avoid fighting 
major trends) and volume confirmation, this captures reversals while 
filtering false signals. Works in BOTH bull (buy BB lower bounces) and bear 
(buy BB upper bounces in rallies OR short BB upper in breakdowns).

TIMEFRAME: 12h primary
HTF: 1d for trend alignment
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_bb_bounce_1d_trend_v1"
timeframe = "12h"
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

def calculate_bollinger_bands(close, period=20, num_std=2.0):
    """Bollinger Bands - returns middle, upper, lower"""
    n = len(close)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower

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
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend alignment
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Bollinger Bands (20, 2.0)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, num_std=2.0)
    
    # RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    # BB Bandwidth for regime detection
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_width_ratio = bb_width / (bb_width_ma + 1e-10)
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Position in BB (0 = lower, 1 = middle, 2 = upper)
    bb_range = bb_upper - bb_lower
    bb_position = (close - bb_lower) / (bb_range + 1e-10)
    bb_position = np.clip(bb_position, 0, 1)
    
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
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
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
        
        # Current values
        close_i = close[i]
        rsi_val = rsi[i]
        vol_val = vol_ratio[i]
        bb_pos = bb_position[i]
        
        # 1d trend: price above HMA = bullish
        price_above_1d_hma = close_i > hma_1d_aligned[i]
        
        # BB regime: narrow bands = potential breakout, wide = trending
        # For mean reversion, we want moderate width (not squeeze, not expanded)
        bb_regime_ok = (0.5 <= bb_width_ratio[i] <= 1.5) if not np.isnan(bb_width_ratio[i]) else True
        
        # Volume confirmation
        vol_confirm = vol_val > 1.15
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === LONG ENTRY: Price at/near lower BB + oversold RSI ===
            # Trigger: BB position < 0.2 (near lower band) + RSI < 40 + volume spike + 1d bullish
            long_trigger = (bb_pos < 0.20) and (rsi_val < 40) and vol_confirm and price_above_1d_hma
            
            if long_trigger:
                desired_signal = SIZE
            else:
                # === SHORT ENTRY: Price at/near upper BB + overbought RSI ===
                # Trigger: BB position > 0.80 (near upper band) + RSI > 60 + volume spike
                # In bear market, shorting upper BB works better
                short_trigger = (bb_pos > 0.80) and (rsi_val > 60) and vol_confirm
                
                if short_trigger:
                    desired_signal = -SIZE
                else:
                    # Alternative: Breakout from squeeze in trend direction
                    # Narrow bands + close breaks above upper = short squeeze
                    squeeze_long = (bb_width_ratio[i] < 0.7) and (close_i > bb_upper[i]) and price_above_1d_hma and vol_confirm
                    squeeze_short = (bb_width_ratio[i] < 0.7) and (close_i < bb_lower[i]) and not price_above_1d_hma and vol_confirm
                    
                    if squeeze_long:
                        desired_signal = SIZE
                    elif squeeze_short:
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
        
        # === TAKE PROFIT: Mean reversion target ===
        # For longs: take profit when RSI normalizes or price reaches middle BB
        # For shorts: take profit when RSI normalizes or price reaches middle BB
        tp_triggered = False
        
        if in_position and position_side > 0:
            # Long TP: RSI > 50 (normalized) OR price > middle BB
            if rsi_val > 50:
                tp_triggered = True
            if close_i > bb_mid[i]:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # Short TP: RSI < 50 (normalized) OR price < middle BB
            if rsi_val < 50:
                tp_triggered = True
            if close_i < bb_mid[i]:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close_i
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