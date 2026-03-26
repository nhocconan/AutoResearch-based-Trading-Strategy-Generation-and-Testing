#!/usr/bin/env python3
"""
Experiment #714: 1d Primary + 1w HTF — Choppiness Regime + Dual Strategy

Hypothesis: 1d timeframe with 1w HTF bias provides optimal trade frequency (20-50/year)
with reduced noise. Using Choppiness Index to switch between:
- Trend regime (CHOP < 38.2): Donchian breakout + 1w HMA trend follow
- Range regime (CHOP > 61.8): Connors RSI mean reversion

Key innovations:
1. Choppiness Index(14) regime detection - switches strategy logic
2. 1w HMA(21) for HTF bias confirmation
3. Donchian(20) breakouts in trend regime
4. Connors RSI (RSI2 + RSI_Streak + PercentRank) for mean reversion
5. ATR(14) 2.5x trailing stop
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure trades):
- LONG trend: 1w HMA bull + (CHOP<45 OR Donchian breakout)
- LONG mean revert: 1w HMA bull + CRSI<20
- SHORT trend: 1w HMA bear + (CHOP<45 OR Donchian breakdown)
- SHORT mean revert: 1w HMA bear + CRSI>80

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies trending vs ranging markets"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100):
    """Connors RSI - combines momentum, streak, and percentile rank"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(2) - very short term momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_2 = np.zeros(n)
    rsi_2[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi_2[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_2[i] = 100.0
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n)
    streak[0] = 1 if close[0] >= close[0] else -1
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like scale
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        streak_sum = 0.0
        count = 0
        for j in range(i-streak_period+1, i+1):
            if streak[j] > 0:
                streak_sum += 100.0 / (streak[j] + 1)
                count += 1
            elif streak[j] < 0:
                streak_sum += 100.0 / (abs(streak[j]) + 1)
                count += 1
        if count > 0:
            rsi_streak[i] = streak_sum / count
    
    # Percentile Rank(100) - where current close ranks in last 100 days
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # CRSI = average of three components
    crsi = (rsi_2 + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_21 = calculate_hma(close, period=21)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(chop[i]):
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
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # Lower threshold = more likely trending
        # Higher threshold = more likely ranging
        trend_regime = chop[i] < 45.0  # Loose threshold for more trades
        range_regime = chop[i] > 55.0  # Loose threshold for more trades
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = False
        donchian_breakout_short = False
        if not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            donchian_breakout_long = close[i] > donchian_upper[i-1]
            donchian_breakout_short = close[i] < donchian_lower[i-1]
        
        # === CONNORS RSI EXTREMES (LOOSE for more trades) ===
        crsi_oversold = crsi[i] < 25.0  # Was 15, now 25 for more trades
        crsi_overbought = crsi[i] > 75.0  # Was 85, now 75 for more trades
        
        # === HMA TREND CONFIRMATION ===
        hma_bull = close[i] > hma_21[i]
        hma_bear = close[i] < hma_21[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: Trend regime + HTF bull + breakout OR HMA bull
        if trend_regime and htf_1w_bull:
            if donchian_breakout_long or hma_bull:
                desired_signal = SIZE_STRONG
        # LONG: Range regime + HTF bull + CRSI oversold
        elif range_regime and htf_1w_bull and crsi_oversold:
            desired_signal = SIZE_BASE
        # LONG: HTF bull + CRSI very oversold (any regime)
        elif htf_1w_bull and crsi[i] < 15.0:
            desired_signal = SIZE_BASE
        
        # SHORT: Trend regime + HTF bear + breakdown OR HMA bear
        elif trend_regime and htf_1w_bear:
            if donchian_breakout_short or hma_bear:
                desired_signal = -SIZE_STRONG
        # SHORT: Range regime + HTF bear + CRSI overbought
        elif range_regime and htf_1w_bear and crsi_overbought:
            desired_signal = -SIZE_BASE
        # SHORT: HTF bear + CRSI very overbought (any regime)
        elif htf_1w_bear and crsi[i] > 85.0:
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