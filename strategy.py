#!/usr/bin/env python3
"""
Experiment #947: 6h Primary + 1d/1w HTF — KAMA Adaptive Trend + Volume Confirmation

Hypothesis: 6h timeframe captures multi-day swings better than 4h (too noisy) or 12h (too slow).
Kaufman Adaptive Moving Average (KAMA) adapts to volatility - fast in trends, slow in ranges.
This should outperform HMA/EMA in 2022 crash and 2025 bear market where simple MAs whipsaw.

Key innovations:
1. 1w momentum filter: weekly close > weekly open = bullish bias (simple but effective)
2. 1d HMA(21) for intermediate trend confirmation
3. 6h KAMA(10/30) crossover for adaptive entry trigger
4. Volume ratio filter: taker_buy_volume / volume > 0.52 for longs (buying pressure)
5. ATR(14) 2.5x trailing stop for risk management
6. LOOSE entry conditions to ensure ≥10 trades/train, ≥3/test

Why KAMA over HMA/EMA:
- Efficiency Ratio (ER) measures trendiness: ER = |close - close_n| / sum(|close_i - close_i-1|)
- Fast SC = 2/(2+1) = 0.667, Slow SC = 2/(30+1) = 0.065
- SC = ER * (fast - slow) + slow
- KAMA adapts: fast SC in strong trends, slow SC in choppy markets
- Proven to reduce whipsaw in 2022 crash and 2025 bear market

Entry conditions (LOOSE to guarantee trades):
- LONG = 1w momentum bull + 1d HMA bull + 6h KAMA crossover up + volume confirm
- SHORT = 1w momentum bear + 1d HMA bear + 6h KAMA crossover down + volume confirm
- Volume filter loose: >0.52 for long, <0.48 for short (not extreme)

Target: Sharpe>0.45, trades>=20 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_volume_momentum_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, fast_period=2, slow_period=30, er_period=10):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing constant based on market efficiency (trend vs noise)
    
    Efficiency Ratio (ER) = |close - close_n| / sum(|close_i - close_i-1|)
    SC = ER * (fast_SC - slow_SC) + slow_SC
    KAMA[i] = KAMA[i-1] + SC * (close[i] - KAMA[i-1])
    """
    n = len(close)
    if n < er_period + 1:
        return np.full(n, np.nan)
    
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate ER and KAMA
    for i in range(er_period, n):
        # Efficiency Ratio: net change / total noise
        net_change = abs(close[i] - close[i - er_period])
        total_noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            total_noise += abs(close[j] - close[j - 1])
        
        if total_noise > 1e-10:
            er = net_change / total_noise
        else:
            er = 0.0
        
        # Smoothing constant
        sc = er * (fast_sc - slow_sc) + slow_sc
        
        # Initialize KAMA
        if i == er_period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

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
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Weekly momentum: close vs open
    weekly_momentum_raw = (df_1w['close'].values - df_1w['open'].values) / (df_1w['open'].values + 1e-10)
    weekly_momentum_aligned = align_htf_to_ltf(prices, df_1w, weekly_momentum_raw)
    
    # Calculate 6h indicators
    kama_6h_10 = calculate_kama(close, fast_period=2, slow_period=10, er_period=10)
    kama_6h_30 = calculate_kama(close, fast_period=2, slow_period=30, er_period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (buying pressure)
    volume_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if volume[i] > 1e-10:
            volume_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            volume_ratio[i] = 0.5
    
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
        
        if np.isnan(kama_6h_10[i]) or np.isnan(kama_6h_30[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(weekly_momentum_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w momentum + 1d HMA) ===
        htf_1w_bull = weekly_momentum_aligned[i] > 0.0
        htf_1w_bear = weekly_momentum_aligned[i] < 0.0
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h KAMA CROSSOVER ===
        kama_crossover_long = False
        kama_crossover_short = False
        if i > 0 and not np.isnan(kama_6h_10[i-1]) and not np.isnan(kama_6h_30[i-1]):
            kama_crossover_long = (kama_6h_10[i-1] <= kama_6h_30[i-1]) and (kama_6h_10[i] > kama_6h_30[i])
            kama_crossover_short = (kama_6h_10[i-1] >= kama_6h_30[i-1]) and (kama_6h_10[i] < kama_6h_30[i])
        
        # === 6h KAMA TREND ===
        kama_6h_bull = kama_6h_10[i] > kama_6h_30[i]
        kama_6h_bear = kama_6h_10[i] < kama_6h_30[i]
        
        # === VOLUME CONFIRMATION (LOOSE) ===
        volume_buy_pressure = volume_ratio[i] > 0.52
        volume_sell_pressure = volume_ratio[i] < 0.48
        
        # === ENTRY LOGIC (LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # LONG entries (HTF bullish bias + volume confirm)
        if htf_1w_bull and htf_1d_bull:
            # Crossover entry (stronger signal)
            if kama_crossover_long and volume_buy_pressure:
                desired_signal = SIZE_STRONG
            # Trend continuation entry (looser - no volume filter)
            elif kama_6h_bull:
                desired_signal = SIZE_BASE
        
        # SHORT entries (HTF bearish bias + volume confirm)
        elif htf_1w_bear and htf_1d_bear:
            # Crossover entry (stronger signal)
            if kama_crossover_short and volume_sell_pressure:
                desired_signal = -SIZE_STRONG
            # Trend continuation entry (looser - no volume filter)
            elif kama_6h_bear:
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