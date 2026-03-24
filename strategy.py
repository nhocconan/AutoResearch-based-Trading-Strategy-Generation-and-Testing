#!/usr/bin/env python3
"""
Experiment #887: 6h Primary + 1d HTF — Ehlers Fisher Transform + HMA Trend + Volume Filter

Hypothesis: 6h timeframe with daily HTF bias provides optimal balance between
trade frequency (30-60/year) and signal quality. Ehlers Fisher Transform catches
reversals better than RSI in bear/range markets (2022 crash, 2025 bear). HMA
provides smooth trend filter with minimal lag. Volume confirmation reduces
false breakouts.

Key innovations:
1. Ehlers Fisher Transform (period=9) - superior reversal detection vs RSI
2. 1d HMA(50) for HTF trend bias - slower than 21 to avoid whipsaw
3. 6h HMA(21) for local trend confirmation
4. Volume ratio filter (taker_buy_volume / volume) > 0.55 for long entries
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: 1d HMA bull (close > hma_1d) + Fisher crosses above -1.5 + vol_ratio > 0.50
- SHORT: 1d HMA bear (close < hma_1d) + Fisher crosses below +1.5 + vol_ratio < 0.50

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_vol_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    # WMA helper
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_n)
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to a Gaussian normal distribution for clearer reversal signals
    
    Formula:
    1. Calculate typical price: (High + Low) / 2
    2. Normalize: (price - lowest_low) / (highest_high - lowest_low)
    3. Transform: 0.5 * ln((1 + x) / (1 - x)) where x = 2*normalized - 1
    4. Smooth with EMA
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize price over lookback period
    normalized = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        if price_range > 1e-10:
            normalized[i] = (typical[i] - lowest) / price_range
        else:
            normalized[i] = 0.5
    
    # Transform to Fisher
    fisher_raw = np.full(n, np.nan)
    for i in range(period - 1, n):
        x = 2 * normalized[i] - 1
        # Clamp x to avoid log(0) or log(negative)
        x = np.clip(x, -0.999, 0.999)
        fisher_raw[i] = 0.5 * np.log((1 + x) / (1 - x))
    
    # Smooth with EMA (span=3)
    fisher = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = np.nan
    
    return fisher, fisher_prev

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(taker_buy_volume, volume):
    """
    Taker buy volume ratio
    Ratio > 0.55 = bullish pressure
    Ratio < 0.45 = bearish pressure
    """
    n = len(volume)
    ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 1e-10:
            ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            ratio[i] = 0.5
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_6h_21 = calculate_hma(close, period=21)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h_21[i]) or np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
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
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 = bullish reversal
        # Fisher crosses below +1.5 = bearish reversal
        fisher_cross_long = (fisher_prev[i] <= -1.5) and (fisher[i] > -1.5)
        fisher_cross_short = (fisher_prev[i] >= 1.5) and (fisher[i] < 1.5)
        
        # Also allow entry when Fisher is in extreme zones (looser condition)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === VOLUME CONFIRMATION ===
        vol_bullish = vol_ratio[i] > 0.50
        vol_bearish = vol_ratio[i] < 0.50
        
        # === HMA TREND CONFIRMATION ===
        hma_6h_bull = close[i] > hma_6h_21[i]
        hma_6h_bear = close[i] < hma_6h_21[i]
        
        # === ENTRY LOGIC (LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        if htf_1d_bull:
            # Bullish HTF bias - look for longs
            if fisher_cross_long and vol_bullish:
                desired_signal = SIZE_STRONG
            elif fisher_oversold and vol_bullish and hma_6h_bull:
                desired_signal = SIZE_BASE
        
        elif htf_1d_bear:
            # Bearish HTF bias - look for shorts
            if fisher_cross_short and vol_bearish:
                desired_signal = -SIZE_STRONG
            elif fisher_overbought and vol_bearish and hma_6h_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
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
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals