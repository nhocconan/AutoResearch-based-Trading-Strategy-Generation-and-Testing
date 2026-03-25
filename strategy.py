#!/usr/bin/env python3
"""
Experiment #1611: 6h Primary + 1d/1w HTF — Connors RSI + KAMA Adaptive Trend + Choppiness Regime

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Connors RSI (CRSI) provides
superior mean-reversion signals vs standard RSI, especially in bear/range markets (2022-2024).
Combined with KAMA (Kaufman Adaptive Moving Average) for trend filtering and Choppiness Index
for regime detection, this should capture both trend continuations and counter-trend bounces.

Key innovations vs previous 6h attempts:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than RSI(14), catches reversals faster
   - Proven 75% win rate in mean-reversion setups
   - LONG: CRSI<20, SHORT: CRSI>80 (loose thresholds for trade count)
2. KAMA ADAPTIVE TREND: ER (Efficiency Ratio) adjusts smoothing based on market noise
   - Fast in trends, slow in chop - perfect for 6h regime changes
   - LONG: price>KAMA + KAMA slope up, SHORT: price<KAMA + KAMA slope down
3. CHOPPINESS REGIME: CHOP>61 = range (mean revert), CHOP<38 = trend (follow KAMA)
4. 1d + 1w HMA BIAS: Ultra-long-term direction filter (prevents major counter-trend)
5. VOLUME SPIKE CONFIRMATION: volume > 1.5x 20-period avg for breakout entries

Why this should beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- CRSI superior to RSI for mean-reversion in bear markets (2022-2024 data)
- KAMA adapts to volatility better than fixed EMA/HMA
- 6h TF = sweet spot between 4h (proven) and 12h (slower signals)
- LOOSE entry thresholds guarantee ≥30 trades/train

Entry logic (LOOSE to guarantee trades):
- LONG trend: 1d_HMA bullish + CHOP<38 + CRSI<30 + price>KAMA + KAMA slope up
- SHORT trend: 1d_HMA bearish + CHOP<38 + CRSI>70 + price<KAMA + KAMA slope down
- LONG range: CHOP>61 + CRSI<20 + price<BB_lower (mean reversion)
- SHORT range: CHOP>61 + CRSI>80 + price>BB_upper (mean reversion)
- NEUTRAL: 1w_HMA bias + CRSI extremes only

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_kama_chop_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - composite mean-reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Proven superior to standard RSI for short-term reversals.
    LONG: CRSI < 20, SHORT: CRSI > 80
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measures consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 and streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 and streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak_window = streak[i - streak_period + 1:i + 1]
        gains = np.sum(np.maximum(streak_window, 0))
        losses = np.abs(np.sum(np.minimum(streak_window, 0)))
        if losses > 0:
            rs = gains / losses
            streak_rsi[i] = 100 - (100 / (1 + rs))
        elif gains > 0:
            streak_rsi[i] = 100
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - current close vs last 100 closes
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    
    ER (Efficiency Ratio) = |price change| / sum of individual changes
    High ER = trending (use fast SC), Low ER = choppy (use slow SC)
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with SMA
    kama[er_period] = np.nanmean(close[:er_period + 1])
    
    # Calculate adaptive KAMA
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama, er

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_volume_ratio(volume, period=20):
    """Current volume vs average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
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
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    kama, er = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
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
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === TREND DIRECTION (1d and 1w HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND FILTER ===
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # KAMA slope (compare to 3 bars ago)
        kama_slope_up = False
        kama_slope_down = False
        if i >= 3 and not np.isnan(kama[i-3]):
            kama_slope_up = kama[i] > kama[i-3]
            kama_slope_down = kama[i] < kama[i-3]
        
        # === CONNORS RSI SIGNALS (LOOSE thresholds for trade count) ===
        crsi_oversold = crsi[i] < 30  # LONG signal
        crsi_overbought = crsi[i] > 70  # SHORT signal
        crsi_extreme_low = crsi[i] < 20  # Strong LONG
        crsi_extreme_high = crsi[i] > 80  # Strong SHORT
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.5 if not np.isnan(vol_ratio[i]) else False
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.005
        bb_touch_upper = close[i] >= bb_upper[i] * 0.995
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow KAMA direction + CRSI pullback + 1d bias
        if is_trend_regime:
            # LONG: 1d bullish + price>KAMA + KAMA slope up + CRSI oversold
            if price_above_1d and price_above_kama and kama_slope_up and crsi_oversold:
                desired_signal = SIZE_STRONG if vol_confirmed else SIZE_BASE
            
            # SHORT: 1d bearish + price<KAMA + KAMA slope down + CRSI overbought
            elif price_below_1d and price_below_kama and kama_slope_down and crsi_overbought:
                desired_signal = -SIZE_STRONG if vol_confirmed else -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at Bollinger bands + CRSI extremes
        elif is_range_regime:
            # LONG: CRSI extreme low + price at BB lower
            if crsi_extreme_low and bb_touch_lower:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI extreme high + price at BB upper
            elif crsi_extreme_high and bb_touch_upper:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Use 1w HMA for ultra-long bias + CRSI only
        else:
            # LONG: 1w bullish + CRSI oversold (not extreme)
            if price_above_1w and crsi_oversold and not crsi_extreme_high:
                desired_signal = SIZE_BASE
            
            # SHORT: 1w bearish + CRSI overbought (not extreme)
            elif price_below_1w and crsi_overbought and not crsi_extreme_low:
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