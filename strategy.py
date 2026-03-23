#!/usr/bin/env python3
"""
Experiment #215: 1h Primary + 4h/1d HTF — Asymmetric Trend + Vol Spike Reversion

Hypothesis: Previous CHOP+CRSI regime strategies failed because regime detection
creates whipsaws. Instead, use ASYMMETRIC logic: different entry rules for long vs short
based on HTF trend confirmation. Long only in bull macro (4h HMA + 1d HMA bullish),
short only in bear macro. Add VOLATILITY SPIKE filter (ATR7/ATR30 > 1.8) to catch
panic reversals. Session filter (8-20 UTC) + volume filter reduce trade frequency.

Key innovations:
1. Asymmetric entries: long pullbacks in bull, short bounces in bear (not symmetric)
2. Vol spike reversion: ATR ratio > 1.8 + BB extreme = high-probability reversal
3. 4h HMA for intermediate trend, 1d HMA for macro bias (dual HTF confirmation)
4. 1h Fisher Transform for precise entry timing within HTF trend
5. Strict session filter (8-20 UTC) to avoid low-liquidity whipsaws
6. Target: 40-70 trades/year on 1h (much lower than typical 1h strategies)

Position sizing: 0.0, ±0.20, ±0.25 (discrete, conservative for 1h TF)
Stoploss: 2.5 * ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_asymm_vol_fisher_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Transforms price into Gaussian-like distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    # Calculate typical price
    tp = (high + low + close) / 3.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        # Normalize price to -1 to +1
        normalized = 2.0 * (tp[i] - lowest) / range_val - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        # Smooth fisher
        if i > period:
            fisher[i] = 0.7 * fisher[i] + 0.3 * fisher[i-1]
        
        trigger[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, trigger

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        change = np.abs(close[i] - close[i-period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility > 1e-10:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    kama_10 = calculate_kama(close, period=10)
    
    vol_ma = calculate_volume_ma(volume, period=20)
    
    # Calculate 4h HMA for intermediate trend (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro bias (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h KAMA for trend confirmation
    kama_4h_raw = calculate_kama(df_4h['close'].values, period=10)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.25
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(fisher[i]) or np.isnan(bb_upper[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(kama_4h_aligned[i]):
            continue
        
        # === VOLATILITY SPIKE FILTER ===
        # ATR7/ATR30 ratio > 1.8 indicates volatility expansion (panic/opportunity)
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = atr_ratio > 1.8
        vol_normal = atr_ratio < 1.3
        
        # === PRICE POSITION IN BB ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        bb_extreme_low = bb_position < 0.15  # Near lower band
        bb_extreme_high = bb_position > 0.85  # Near upper band
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        vol_above_avg = volume[i] > 0.8 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # === HTF MACRO BIAS ===
        # 4h HMA for intermediate trend
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # 1d HMA for macro bias
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # 4h KAMA for trend confirmation
        price_above_kama_4h = close[i] > kama_4h_aligned[i]
        price_below_kama_4h = close[i] < kama_4h_aligned[i]
        
        # === MACRO REGIME ===
        bull_macro = price_above_hma_4h and price_above_hma_1d and price_above_kama_4h
        bear_macro = price_below_hma_4h and price_below_hma_1d and price_below_kama_4h
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] < -1.5  # Cross above -1.5
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] > 1.5  # Cross below +1.5
        
        # === ENTRY LOGIC (ASYMMETRIC) ===
        new_signal = 0.0
        
        # LONG ENTRY: Only in bull macro OR vol spike reversion
        if bull_macro:
            # Trend-following long: Fisher long signal + in session + volume
            if fisher_long and in_session and vol_above_avg:
                new_signal = POSITION_SIZE_FULL
        elif vol_spike and bb_extreme_low:
            # Vol spike reversion long (counter-trend, smaller size)
            if in_session and vol_above_avg:
                new_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: Only in bear macro OR vol spike reversion
        elif bear_macro:
            # Trend-following short: Fisher short signal + in session + volume
            if fisher_short and in_session and vol_above_avg:
                new_signal = -POSITION_SIZE_FULL
        elif vol_spike and bb_extreme_high:
            # Vol spike reversion short (counter-trend, smaller size)
            if in_session and vol_above_avg:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if macro still bullish or Fisher not reversed
                if bull_macro or (fisher[i] > -1.0 and not bb_extreme_high):
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if macro still bearish or Fisher not reversed
                if bear_macro or (fisher[i] < 1.0 and not bb_extreme_low):
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === MACRO REVERSAL EXIT ===
        # Exit long if macro turns bearish
        if in_position and position_side > 0 and bear_macro:
            new_signal = 0.0
        
        # Exit short if macro turns bullish
        if in_position and position_side < 0 and bull_macro:
            new_signal = 0.0
        
        # Exit on Fisher reversal
        if in_position and position_side > 0 and fisher_short:
            new_signal = 0.0
        
        if in_position and position_side < 0 and fisher_long:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals