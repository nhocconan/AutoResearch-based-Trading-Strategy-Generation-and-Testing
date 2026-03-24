#!/usr/bin/env python3
"""
Experiment #899: 1h Primary + 4h/12h HTF — HMA Trend + Fisher Transform + Volume

Hypothesis: 1h timeframe with 4h/12h HTF bias provides optimal trade frequency
(40-80 trades/year). Hull Moving Average reduces lag for trend detection.
Ehlers Fisher Transform excels at catching reversals in bear/range markets
(2022 crash, 2025 bear). Volume confirmation filters false signals.

Key innovations:
1. 12h HMA(48) for macro trend bias - very smooth, low whipsaw
2. 4h HMA(21) for intermediate trend confirmation
3. Ehlers Fisher Transform(9) for entry timing - superior to RSI for reversals
4. Taker buy volume ratio for confirmation (>0.55 = bullish pressure)
5. Session filter: 08-20 UTC (high liquidity, avoid Asian chop)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.20, ±0.30

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: 12h HMA bull + 4h HMA bull + Fisher < -1.0 + volume ratio > 0.52
- SHORT: 12h HMA bear + 4h HMA bear + Fisher > +1.0 + volume ratio < 0.48
- Session: only enter 08-20 UTC (avoid low-liquidity whipsaws)

Target: Sharpe>0.45, trades>=40/train, trades>=5/test, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_fisher_volume_session_4h12h_v1"
timeframe = "1h"
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
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
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
    Normalizes price to -1 to +1 range, excellent for catching reversals
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = EMA of normalized price
    """
    n = len(high)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + np.roll(high + low, 1)) / 4.0
    typical[0] = (high[0] + low[0]) / 2.0
    
    # Normalize to -1 to +1 range using highest high / lowest low over period
    fisher = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize price to 0-1 range, then to -0.99 to +0.99
        normalized = 2.0 * (typical[i] - lowest) / price_range - 1.0
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # EMA of normalized value
        if i == period:
            x_value = normalized
        else:
            x_value = 0.7 * normalized + 0.3 * x_prev
        
        x_prev = x_value
        
        # Fisher transform
        if abs(x_value) < 0.999:
            fisher[i] = 0.5 * np.log((1 + x_value) / (1 - x_value))
    
    return fisher

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
    Taker Buy Volume Ratio
    Ratio > 0.55 = bullish pressure, < 0.45 = bearish pressure
    """
    n = len(volume)
    ratio = np.full(n, np.nan)
    
    for i in range(n):
        if volume[i] > 1e-10:
            ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            ratio[i] = 0.5
    
    return ratio

def get_hour_from_open_time(open_time_array):
    """
    Extract UTC hour from open_time (milliseconds timestamp)
    Returns array of hours (0-23)
    """
    # Convert milliseconds to seconds, then to datetime
    timestamps = open_time_array / 1000.0
    hours = np.zeros(len(timestamps), dtype=np.int32)
    
    for i, ts in enumerate(timestamps):
        # Get hour from UTC timestamp
        hours[i] = int((ts % 86400) / 3600)
    
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=48)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    fisher = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        in_session = 8 <= hours[i] <= 20
        
        # === HTF BIAS (12h HMA) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.0  # Reversal long signal
        fisher_overbought = fisher[i] > +1.0  # Reversal short signal
        fisher_neutral_long = fisher[i] < -0.5  # Loose long
        fisher_neutral_short = fisher[i] > +0.5  # Loose short
        
        # === VOLUME CONFIRMATION ===
        vol_bullish = vol_ratio[i] > 0.52
        vol_bearish = vol_ratio[i] < 0.48
        
        # === ENTRY LOGIC (3+ CONFLUENCE) ===
        desired_signal = 0.0
        
        # LONG: 12h bull + 4h bull + Fisher oversold + volume bullish + session
        if htf_12h_bull and htf_4h_bull and in_session:
            if fisher_oversold and vol_bullish:
                desired_signal = SIZE_STRONG
            elif fisher_neutral_long and vol_bullish:
                desired_signal = SIZE_BASE
        
        # SHORT: 12h bear + 4h bear + Fisher overbought + volume bearish + session
        elif htf_12h_bear and htf_4h_bear and in_session:
            if fisher_overbought and vol_bearish:
                desired_signal = -SIZE_STRONG
            elif fisher_neutral_short and vol_bearish:
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