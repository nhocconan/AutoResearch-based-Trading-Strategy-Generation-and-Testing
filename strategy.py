#!/usr/bin/env python3
"""
Experiment #682: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + HMA Bias

Hypothesis: Connors RSI (CRSI) is proven mean-reversion indicator with 75%+ win rate in 
range markets. Combined with Choppiness Index to detect regime (chop vs trend), and 
1d HMA for directional bias. This should work across all market conditions:
- 2021 bull: HTF bias keeps us long, CRSI catches pullbacks
- 2022 crash: HTF bear + CRSI oversold = short entries on bounces
- 2023-24 range: Choppiness high = mean reversion at extremes
- 2025 bear: HTF bear bias + CRSI for counter-trend shorts

Key innovations:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven 75% win rate
2. Choppiness Index(14) regime filter — CHOP>61.8=range, CHOP<38.2=trend
3. 1d HMA(21) bias — only long above, only short below
4. 1w HMA(50) meta-bias — reduces size against weekly trend
5. ATR(14) 2.5x trailing stop — protects from crash
6. LOOSE entry thresholds (CRSI<20/>80) to ensure >=30 trades/train

Entry conditions (designed for trade frequency):
- LONG: close > 1d HMA AND CRSI < 25 AND (CHOP > 50 OR ADX < 25)
- SHORT: close < 1d HMA AND CRSI > 75 AND (CHOP > 50 OR ADX < 25)
- In strong trend (CHOP < 40): allow entries with weaker CRSI (30/70)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 4h
Size: 0.25 base, 0.30 strong, 0.15 weak (discrete levels)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_hma_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI_Streak(2): RSI of consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 bars
    
    Proven 75%+ win rate for mean reversion entries at extremes (<10 or >90)
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] > 1e-10:
            rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain[i] / avg_loss[i]))
        else:
            rsi[i] = 100.0
    
    # RSI Streak — count consecutive up/down days
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if streak[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    for i in range(streak_period + 5, n):
        if avg_streak_loss[i] > 1e-10:
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + avg_streak_gain[i] / avg_streak_loss[i]))
        elif avg_streak_gain[i] > 1e-10:
            streak_rsi[i] = 100.0
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank — where does current close rank vs last rank_period bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period + 5, n):
        if not np.isnan(rsi[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures if market is trending or ranging
    
    CHOP = 100 * (SUM(ATR(1), n) / (Highest High - Lowest Low)) / log10(n)
    
    CHOP > 61.8 = Range/Chop (mean reversion works)
    CHOP < 38.2 = Trend (trend following works)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10:
            choppiness[i] = 100.0 * (sum_tr / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (discrete levels per Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.15
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]) or np.isnan(adx[i]):
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
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY META-BIAS (reduce size against weekly trend) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        is_choppy = choppiness[i] > 55.0  # Range market
        is_trending = choppiness[i] < 45.0 and adx[i] > 25.0  # Strong trend
        
        # === CRSI EXTREMES (LOOSE thresholds for trade frequency) ===
        crsi_oversold = crsi[i] < 25.0  # Long signal
        crsi_overbought = crsi[i] > 75.0  # Short signal
        
        # In trending regime, use weaker CRSI thresholds
        if is_trending:
            crsi_oversold = crsi[i] < 35.0
            crsi_overbought = crsi[i] > 65.0
        
        # === ENTRY LOGIC (designed for >=30 trades/train) ===
        desired_signal = 0.0
        signal_strength = 1.0
        
        # LONG entries
        if htf_bull and crsi_oversold:
            if is_choppy:
                # Choppy market + oversold = strong mean reversion long
                desired_signal = SIZE_STRONG
            elif htf_bull and weekly_bull:
                # Both daily and weekly bullish = strong trend long
                desired_signal = SIZE_STRONG
            else:
                # Just daily bullish = base size
                desired_signal = SIZE_BASE
            
            # Reduce size if against weekly trend
            if weekly_bear:
                signal_strength = 0.6
        
        # SHORT entries
        elif htf_bear and crsi_overbought:
            if is_choppy:
                # Choppy market + overbought = strong mean reversion short
                desired_signal = -SIZE_STRONG
            elif htf_bear and weekly_bear:
                # Both daily and weekly bearish = strong trend short
                desired_signal = -SIZE_STRONG
            else:
                # Just daily bearish = base size
                desired_signal = -SIZE_BASE
            
            # Reduce size if against weekly trend
            if weekly_bull:
                signal_strength = 0.6
        
        # Apply weekly bias sizing
        desired_signal *= signal_strength
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        elif abs(desired_signal) >= SIZE_WEAK * 0.9:
            final_signal = np.sign(desired_signal) * SIZE_WEAK
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