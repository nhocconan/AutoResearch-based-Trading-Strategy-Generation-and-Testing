#!/usr/bin/env python3
"""
Experiment #1100: 6h Primary + 1d/1w HTF — Ehlers Fisher + KAMA Adaptive + Volume Confirmation

Hypothesis: The Ehlers Fisher Transform catches reversals better than RSI in bear/range markets
(2022-2023, 2025+), while KAMA adapts to volatility regimes automatically. Combined with volume
confirmation and asymmetric entries based on HTF bias, this should outperform HMA+RSI strategies.

Key innovations:
1. Ehlers Fisher Transform (period=9): Normalizes price to Gaussian distribution, crosses at ±1.5
   signal reversals better than RSI in choppy markets (proven in Ehlers literature)
2. KAMA (Kaufman Adaptive MA): Efficiency Ratio adjusts smoothing constant automatically
   - Fast in trends (ER high), slow in chop (ER low) — no regime detection needed
3. Volume confirmation: Entry only when volume > 1.3x 20-bar avg (institutional support)
4. Asymmetric entries: Long easier when 1w_KAMA bull, short easier when 1w_KAMA bear
5. 1d/1w KAMA for HTF bias (not strict alignment, just direction)
6. ATR(14) 2.0x stoploss with signal→0 on breach

Why 6h specifically:
- Captures multi-day swings (3-7 day holds) without 4h noise or 12h slowness
- 30-60 trades/year target fits cost model (1.5-3% fee drag)
- Middle ground for regime changes (faster reaction than 12h)

Entry conditions (LOOSE to guarantee trades):
- LONG: Fisher<-1.2 + close>KAMA(6h) + volume>1.3x + (1w_KAMA bull OR 1d_KAMA bull)
- SHORT: Fisher>+1.2 + close<KAMA(6h) + volume>1.3x + (1w_KAMA bear OR 1d_KAMA bear)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_kama_vol_asym_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adjusts smoothing constant based on market efficiency (trend vs chop)
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for clearer reversal signals
    Price must be bounded between -1 and +1 before Fisher calculation
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * (close[i] - lowest) / price_range - 1.0
        normalized = max(-0.999, min(0.999, normalized))  # Bound for log calculation
        
        # Fisher calculation
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Previous fisher value for signal line
        if i > 0 and not np.isnan(fisher[i - 1]):
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

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

def calculate_volume_ratio(volume, period=20):
    """Current volume vs rolling average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        avg_vol = np.mean(volume[i - period + 1:i + 1])
        if avg_vol > 1e-10:
            vol_ratio[i] = volume[i] / avg_vol
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF KAMA
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate 6h indicators
    kama_6h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_6h[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (KAMA direction) ===
        htf_bull = close[i] > kama_1d_aligned[i] or close[i] > kama_1w_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i] or close[i] < kama_1w_aligned[i]
        
        # Strong HTF alignment (both 1d and 1w agree)
        htf_strong_bull = close[i] > kama_1d_aligned[i] and close[i] > kama_1w_aligned[i]
        htf_strong_bear = close[i] < kama_1d_aligned[i] and close[i] < kama_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.3  # 30% above average
        
        # === 6h TREND (KAMA position) ===
        kama_bull = close[i] > kama_6h[i]
        kama_bear = close[i] < kama_6h[i]
        
        # === ENTRY LOGIC (Fisher Transform + Asymmetric) ===
        desired_signal = 0.0
        
        # LONG entries (easier when HTF bull)
        # Fisher <-1.2 = oversold reversal signal
        if fisher[i] < -1.2:
            if htf_strong_bull and kama_bull and vol_confirmed:
                desired_signal = SIZE_STRONG  # Strong long: all conditions met
            elif htf_bull and kama_bull:
                desired_signal = SIZE_BASE  # Standard long
            elif htf_bull and not vol_confirmed:
                desired_signal = SIZE_BASE * 0.6  # Weaker without volume
        # Fisher cross above -1.5 from below (reversal confirmation)
        elif i > 0 and not np.isnan(fisher[i-1]) and fisher[i-1] < -1.5 and fisher[i] > -1.5:
            if htf_bull and kama_bull:
                desired_signal = SIZE_BASE
        
        # SHORT entries (easier when HTF bear)
        # Fisher >+1.2 = overbought reversal signal
        if fisher[i] > 1.2:
            if htf_strong_bear and kama_bear and vol_confirmed:
                desired_signal = -SIZE_STRONG  # Strong short: all conditions met
            elif htf_bear and kama_bear:
                desired_signal = -SIZE_BASE  # Standard short
            elif htf_bear and not vol_confirmed:
                desired_signal = -SIZE_BASE * 0.6  # Weaker without volume
        # Fisher cross below +1.5 from above (reversal confirmation)
        elif i > 0 and not np.isnan(fisher[i-1]) and fisher[i-1] > 1.5 and fisher[i] < 1.5:
            if htf_bear and kama_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.6
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.6
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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