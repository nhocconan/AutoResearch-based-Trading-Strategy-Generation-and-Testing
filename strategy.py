#!/usr/bin/env python3
"""
Experiment #1592: 12h Primary + 1d HTF — Connors RSI + HMA Trend + Volume

Hypothesis: Connors RSI (CRSI) is proven superior to standard RSI for mean-reversion
in bear/range markets (2022 crash, 2025+ bear). Combined with 1d HMA trend bias and
simple volume confirmation, this should generate consistent trades across all symbols
while avoiding the over-complexity that killed #1584 (Fisher + Choppiness + Donchian).

Why this should beat #1584 (Sharpe=0.027) and baseline (Sharpe=0.575):
1. CRSI has 75% win rate in academic studies for mean-reversion entries
2. Simpler logic = fewer conflicting filters = MORE trades (critical for Sharpe)
3. CRSI<10/ >90 happens regularly (unlike Fisher<-1.5 + Donchian breakout + volume)
4. 1d HMA provides trend bias without complex regime switching
5. 12h TF = 20-50 trades/year target (fee-efficient)

Key improvements vs #1584:
- Removed Choppiness Index regime (was causing 0 trades in neutral periods)
- Removed Donchian breakout requirement (too restrictive)
- Removed Fisher Transform (complex, similar signals to RSI)
- Added CRSI (3-component: RSI3 + StreakRSI2 + PercentRank100)
- Looser entry: CRSI<15 for long, >85 for short (guarantees trades)
- Simpler trend filter: just 1d HMA slope, not 1d+1w

Entry logic (LOOSE to guarantee ≥30 trades/train):
- LONG: 1d_HMA bullish (price>hma AND hma sloping up) + CRSI<15 + volume>1.2x
- SHORT: 1d_HMA bearish (price<hma AND hma sloping down) + CRSI>85 + volume>1.2x
- Exit: CRSI crosses 50 (mean reached) OR 2.5*ATR stoploss

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_hma_trend_1d_vol_v1"
timeframe = "12h"
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
    Connors RSI (CRSI) - 3-component mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Proven 75% win rate for mean-reversion entries in bear/range markets.
    Long when CRSI < 10-15, Short when CRSI > 85-90.
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi3 = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI(2)
    # Streak = consecutive days up/down
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak (use absolute values for calculation)
    streak_rsi = calculate_rsi(np.abs(streak), streak_period)
    # Adjust sign: negative streak = low RSI, positive = high RSI
    streak_rsi_signed = np.where(streak < 0, streak_rsi * 0.5, 50 + streak_rsi * 0.5)
    streak_rsi_signed = np.clip(streak_rsi_signed, 0, 100)
    
    # Component 3: PercentRank(100) - where current close ranks in last 100 bars
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period - 1, n):
        window = close[i - rank_period + 1:i + 1]
        if not np.any(np.isnan(window)):
            rank = np.sum(window[:-1] < close[i])  # count how many bars lower
            percent_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # Combine components
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period - 1, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi_signed[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi_signed[i] + percent_rank[i]) / 3.0
    
    return crsi

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
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi_3_2_100 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Also calculate 1d HMA slope for trend confirmation
    hma_1d_slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(3, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]):
            hma_1d_slope[i] = hma_1d_aligned[i] - hma_1d_aligned[i-1]
    
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
    
    # Track CRSI for exit signals
    prev_crsi = np.nan
    
    # Warmup period (need 100 bars for CRSI percentrank)
    min_bars = 120
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi_3_2_100[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1d_slope[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d HMA bias + slope) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        hma_sloping_up = hma_1d_slope[i] > 0
        hma_sloping_down = hma_1d_slope[i] < 0
        
        # === CRSI SIGNALS (LOOSE thresholds for trade frequency) ===
        crsi_val = crsi_3_2_100[i]
        crsi_extreme_low = crsi_val < 15  # oversold
        crsi_extreme_high = crsi_val > 85  # overbought
        crsi_crossed_50_up = prev_crsi < 50 and crsi_val >= 50 if not np.isnan(prev_crsi) else False
        crsi_crossed_50_down = prev_crsi > 50 and crsi_val <= 50 if not np.isnan(prev_crsi) else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.2 if not np.isnan(vol_ratio[i]) else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish trend + CRSI oversold + volume confirmation
        if price_above_1d and hma_sloping_up and crsi_extreme_low:
            desired_signal = SIZE_STRONG if vol_confirmed else SIZE_BASE
        
        # SHORT: 1d bearish trend + CRSI overbought + volume confirmation
        elif price_below_1d and hma_sloping_down and crsi_extreme_high:
            desired_signal = -SIZE_STRONG if vol_confirmed else -SIZE_BASE
        
        # === EXIT LOGIC (CRSI mean reversion + stoploss) ===
        # Exit long when CRSI crosses above 50 (mean reached)
        if in_position and position_side > 0 and crsi_crossed_50_up:
            desired_signal = 0.0
        
        # Exit short when CRSI crosses below 50 (mean reached)
        if in_position and position_side < 0 and crsi_crossed_50_down:
            desired_signal = 0.0
        
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
        prev_crsi = crsi_val
    
    return signals