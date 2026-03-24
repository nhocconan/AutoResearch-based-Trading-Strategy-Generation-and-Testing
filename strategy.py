#!/usr/bin/env python3
"""
Experiment #602: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Connors RSI (CRSI) provides superior mean reversion signals vs standard RSI,
especially in bear/range markets like 2022-2024. Combined with Choppiness Index for
regime detection and HMA for trend bias, this should generate consistent trades across
BTC/ETH/SOL while maintaining positive Sharpe.

Key improvements over failed #598 (mtf_4h_hma_rsi_simple_1d_v1):
1. Connors RSI instead of standard RSI - 3 components for better signal quality
2. Simpler regime detection - CHOP threshold only (not ADX+CHOP combo that rarely triggers)
3. Looser entry thresholds - CRSI<15/>85 instead of RSI<25/>75 to ensure trades
4. Reduced HTF filter strictness - only need 1d alignment, not both 1d+1w
5. KAMA adaptive trend for better whipsaw protection during volatile periods

Strategy logic:
1. 1d HMA(21) = medium trend bias (only require this, not 1w)
2. 4h HMA(21) = short-term trend direction
3. 4h KAMA(10,2,30) = adaptive trend following (less whipsaw than HMA)
4. 4h Choppiness(14) = regime (CHOP>55 = range, CHOP<45 = trend)
5. 4h Connors RSI = entry timing (CRSI<15 long, CRSI>85 short)
6. ATR(14)*2.5 stoploss on all positions

Regime-adaptive entries:
- RANGE (CHOP>55): Mean revert with CRSI extremes + HMA bias
- TREND (CHOP<45): Follow KAMA direction with HTF confirmation
- TRANSITION (45-55): Reduced size, require stronger signals

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_hma_kama_1d_v2"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Composite of 3 components
    1. RSI(close, 3) - short-term momentum
    2. RSI(streak, 2) - streak strength (consecutive up/down days)
    3. PercentRank(close, 100) - where current price ranks vs last 100
    
    CRSI = (RSI1 + RSI2 + PercentRank) / 3
    Long: CRSI < 10-15 | Short: CRSI > 85-90
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi1 = np.zeros(n)
    rsi1[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi1[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi1[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of Streak(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi2 = np.zeros(n)
    rsi2[:] = np.nan
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi2[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi2[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine components
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi1[i]) and not np.isnan(rsi2[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi1[i] + rsi2[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_4h = calculate_hma(close, period=21)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(hma_4h[i]):
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
        
        # === HTF BIAS (1d only - simpler than 1d+1w) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND ===
        trend_bull = close[i] > hma_4h[i] and close[i] > kama[i]
        trend_bear = close[i] < hma_4h[i] and close[i] < kama[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        
        # === CRSI EXTREMES (looser thresholds for more trades) ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === CRSI RECOVERY ===
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 and not np.isnan(crsi[i-1]) else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 and not np.isnan(crsi[i-1]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with CRSI extremes
        if is_range:
            # Long: CRSI oversold + price above 1d HMA (bullish bias)
            if crsi_extreme_oversold and htf_bull:
                desired_signal = SIZE_STRONG
            elif crsi_oversold and crsi_rising and htf_bull:
                desired_signal = SIZE_BASE
            # Short: CRSI overbought + price below 1d HMA (bearish bias)
            elif crsi_extreme_overbought and htf_bear:
                desired_signal = -SIZE_STRONG
            elif crsi_overbought and crsi_falling and htf_bear:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Follow trend with CRSI pullback
        elif is_trend:
            # Long: Uptrend + CRSI pullback (not extreme)
            if trend_bull and htf_bull and crsi_oversold:
                desired_signal = SIZE_STRONG
            elif trend_bull and htf_bull and crsi[i] < 50.0 and crsi_rising:
                desired_signal = SIZE_BASE
            # Short: Downtrend + CRSI pullback
            elif trend_bear and htf_bear and crsi_overbought:
                desired_signal = -SIZE_STRONG
            elif trend_bear and htf_bear and crsi[i] > 50.0 and crsi_falling:
                desired_signal = -SIZE_BASE
        
        # TRANSITION: Require stronger signals
        else:
            if crsi_extreme_oversold and htf_bull and trend_bull:
                desired_signal = SIZE_BASE
            elif crsi_extreme_overbought and htf_bear and trend_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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