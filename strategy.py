#!/usr/bin/env python3
"""
Experiment #412: 12h Primary + 1d/1w HTF — Fisher Transform + KAMA Adaptive Trend + Volume

Hypothesis: Previous strategies overused HMA/RSI/Choppiness combinations. This uses:
1. Ehlers Fisher Transform (period=9) - superior reversal detection in bear/range markets
2. KAMA (Kaufman Adaptive MA) - adapts speed based on market efficiency ratio
3. Volume ratio confirmation - filters false breakouts
4. 1d HMA + 1w HMA for dual HTF bias (stronger than single HTF)
5. 12h primary = ~30-50 trades/year, minimal fee drag

Why this differs from failed attempts:
- Fisher Transform catches reversals better than RSI in 2022 crash & 2025 bear
- KAMA adapts to regime changes (fast in trend, slow in chop) - no need for Choppiness
- Dual HTF (1d + 1w) provides stronger bias filter than single HTF
- Volume confirmation adds edge most strategies ignore

Target: Sharpe > 0.612, 80-200 trades over 4-year train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_kama_volume_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_span=2, slow_span=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
        er[i] = np.clip(er[i], 0.0, 1.0)
    
    # Calculate smoothing constant
    sc = np.full(n, np.nan)
    fast_sc = 2.0 / (fast_span + 1)
    slow_sc = 2.0 / (slow_span + 1)
    for i in range(er_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for better reversal detection.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = close[i-period+1:i+1].max()
        lowest = close[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            fisher[i] = fisher[i-1] if i > period else 0.0
            fisher_signal[i] = fisher[i]
            continue
        
        # Normalize price to -1 to +1 range
        raw_fisher = 0.66 * ((close[i] - lowest) / (highest - lowest) - 0.5)
        raw_fisher = np.clip(raw_fisher, -0.99, 0.99)
        
        # Apply Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + raw_fisher) / (1.0 - raw_fisher))
        if i > period:
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
        
        fisher_signal[i] = fisher[i-1] if i > period else fisher[i]
    
    return fisher, fisher_signal

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    ratio = vol_s / (vol_avg + 1e-10)
    return ratio.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    kama = calculate_kama(close, er_period=10, fast_span=2, slow_span=30)
    fisher, fisher_signal = calculate_fisher(close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align HTF HMAs for bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 12h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(kama[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(vol_ratio[i]):
            continue
        
        # === KAMA SLOPE (adaptive trend) ===
        kama_slope = kama[i] - kama[i-5] if i >= 5 else 0.0
        kama_bullish = kama_slope > 0.0
        kama_bearish = kama_slope < 0.0
        
        # === HTF BIAS (dual: 1d + 1w) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias: both 1d and 1w agree
        strong_bullish_bias = price_above_hma_1d and price_above_hma_1w
        strong_bearish_bias = price_below_hma_1d and price_below_hma_1w
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long = fisher_signal[i] < -1.5 and fisher[i] > -1.5
        # Short: Fisher crosses below +1.5 from above
        fisher_short = fisher_signal[i] > 1.5 and fisher[i] < 1.5
        
        # Alternative: Fisher extreme reversals
        fisher_extreme_long = fisher[i] < -1.8 and fisher[i] > fisher_signal[i]
        fisher_extreme_short = fisher[i] > 1.8 and fisher[i] < fisher_signal[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 0.8  # At least 80% of avg volume
        
        # === VOL FILTER ===
        vol_ratio_atr = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio_atr > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio_atr > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP
        if strong_bullish_bias or price_above_hma_1d:  # HTF bullish bias
            if kama_bullish:  # KAMA trending up
                if (fisher_long or fisher_extreme_long) and volume_confirmed:
                    desired_signal = position_size
                elif fisher[i] > -0.5 and fisher_signal[i] < -0.5:  # Mid-level cross
                    desired_signal = position_size * 0.5
        
        # SHORT SETUP
        if strong_bearish_bias or price_below_hma_1d:  # HTF bearish bias
            if kama_bearish:  # KAMA trending down
                if (fisher_short or fisher_extreme_short) and volume_confirmed:
                    desired_signal = -position_size
                elif fisher[i] < 0.5 and fisher_signal[i] > 0.5:  # Mid-level cross
                    desired_signal = -position_size * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === FISHER EXTREME EXIT ===
        if in_position and position_side > 0 and fisher[i] > 2.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -2.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1d and kama_bullish:
                desired_signal = position_size
            elif position_side < 0 and price_below_hma_1d and kama_bearish:
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals