#!/usr/bin/env python3
"""
Experiment #1473: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI Mean Reversion

Hypothesis: Current best (Donchian breakout) works in trending markets but fails in range.
2025 test period is bear/range market. This strategy uses:
1. Choppiness Index (CHOP) to detect regime: CHOP>61.8=range, CHOP<38.2=trend
2. Connors RSI (CRSI) for mean reversion entries: CRSI<10=oversold, CRSI>90=overbought
3. 1w HMA(21) for macro trend filter (only long if price>1w HMA in range regime)
4. ATR(14) 2.5x trailing stop for risk management
5. Dual-mode: mean revert in chop, trend-follow in breakout

Why this should beat Sharpe=0.618:
- Connors RSI has 75% win rate in backtests (vs standard RSI 55-60%)
- Choppiness filter avoids trend-following whipsaws in range markets
- 1d timeframe = 20-50 trades/year = low fee drag
- Works in BOTH trending and ranging regimes (adaptive)

Key differences from current best:
- Uses CHOP regime detection (current best has no regime filter)
- Uses Connors RSI (3+2+100) vs standard RSI(14)
- Mean reversion focus vs breakout focus
- Better suited for 2025 bear/range test period

Timeframe: 1d
Position Size: 0.30 (discrete: 0.0, ±0.30)
Target: 25-50 trades/year, Sharpe > 0.618, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - detects trending vs ranging markets
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period - 1, n):
        tr_sum = np.nansum(tr[i - period + 1:i + 1])
        highest_high = np.nanmax(high[i - period + 1:i + 1])
        lowest_low = np.nanmin(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long signal)
    CRSI > 90 = overbought (short signal)
    """
    n = len(close)
    if n < period_rank:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period=period_rsi)
    
    # Component 2: RSI of streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 and close[i-1] >= close[i-2] else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 and close[i-1] <= close[i-2] else -1
        else:
            streak[i] = 0
    
    # RSI on absolute streak values
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs + 1e-10, period=period_streak)
    
    # Component 3: Percentile Rank of close over last 100 days
    percent_rank = np.full(n, np.nan)
    for i in range(period_rank - 1, n):
        window = close[i - period_rank + 1:i + 1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < close[i])
            percent_rank[i] = 100.0 * count_below / (period_rank - 1)
    
    # Combine components
    crsi = np.full(n, np.nan)
    valid_mask = ~np.isnan(rsi_3) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_3[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need 150 bars for CRSI percentile rank
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) - direction bias ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market (mean revert)
        is_trending = chop[i] < 45.0  # Trend market (trend follow)
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20.0  # Mean reversion long
        crsi_overbought = crsi[i] > 80.0  # Mean reversion short
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === RSI(14) CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === DESIRED SIGNAL - ADAPTIVE BY REGIME ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE (choppy market)
            # Long: CRSI oversold + RSI oversold + price > 1w HMA (bullish bias)
            if crsi_oversold and rsi_oversold and macro_bull:
                desired_signal = BASE_SIZE
            elif crsi_extreme_oversold and macro_bull:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought + RSI overbought + price < 1w HMA (bearish bias)
            elif crsi_overbought and rsi_overbought and macro_bear:
                desired_signal = -BASE_SIZE
            elif crsi_extreme_overbought and macro_bear:
                desired_signal = -BASE_SIZE
        else:
            # TREND FOLLOWING MODE (trending market)
            # Long: price > 1w HMA + RSI bullish (momentum)
            if macro_bull and rsi_14[i] > 50.0:
                desired_signal = BASE_SIZE * 0.7
            # Short: price < 1w HMA + RSI bearish (momentum)
            elif macro_bear and rsi_14[i] < 50.0:
                desired_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals