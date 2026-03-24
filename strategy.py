#!/usr/bin/env python3
"""
Experiment #848: 4h Primary + 12h/1d HTF — Connors RSI + Choppiness Regime Switch

Hypothesis: 4h timeframe with 12h HTF bias provides optimal balance between
signal quality and trade frequency. Connors RSI (CRSI) excels at mean reversion
entries in ranging markets, while Choppiness Index detects regime to switch
between mean-revert (chop) and trend-follow (trending) logic.

Key innovations:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - CRSI<10 = oversold long opportunity
   - CRSI>90 = overbought short opportunity
2. Choppiness Index(14) regime detection:
   - CHOP > 55 = ranging → use CRSI mean reversion
   - CHOP < 45 = trending → use HMA breakout
3. 12h HMA(21) for HTF trend bias (only trade with HTF direction)
4. ATR(14) 2.5x trailing stop for risk management
5. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG (chop regime): HTF bull + CRSI<25 + CHOP>50
- LONG (trend regime): HTF bull + price>HMA(21) + RSI(14)>50
- SHORT (chop regime): HTF bear + CRSI>75 + CHOP>50
- SHORT (trend regime): HTF bear + price<HMA(21) + RSI(14)<50

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Short-term momentum
    RSI(Streak): Duration of consecutive up/down days
    PercentRank: Where current return ranks vs last 100 days
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3) - short term momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    rsi_short[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_short[i] = 100.0
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (positive streak = high, negative = low)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        pos_count = np.sum(streak_vals > 0)
        total = len(streak_vals)
        if total > 0:
            streak_rsi[i] = 100.0 * pos_count / total
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank - where current return ranks vs last rank_period returns
    returns = np.diff(close, prepend=close[0]) / (np.where(close != 0, close, 1))
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        window = returns[max(0, i-rank_period+1):i+1]
        current = returns[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100.0 * rank / len(window)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    More responsive than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    def wma(series, span):
        """Weighted Moving Average"""
        result = np.zeros(len(series))
        result[:] = np.nan
        for i in range(span-1, len(series)):
            weights = np.arange(1, span+1)
            window = series[i-span+1:i+1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = period // 2
    if half < 1:
        half = 1
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # Combine
    diff = 2.0 * wma_half - wma_full
    
    # Final WMA with sqrt period
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 55/45 as regime thresholds for 4h TF
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 4h indicators
    hma_21 = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
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
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA) ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_regime = chop_14[i]
        is_choppy = chop_regime > 50.0  # Ranging market
        is_trending = chop_regime < 50.0  # Trending market
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_21[i]
        hma_bear = close[i] < hma_21[i]
        
        # === CRSI EXTREMES (for chop regime) ===
        crsi_oversold = crsi[i] < 25.0  # Mean reversion long
        crsi_overbought = crsi[i] > 75.0  # Mean reversion short
        
        # === RSI FILTER (for trend regime) ===
        rsi_bull = rsi_14[i] > 50.0
        rsi_bear = rsi_14[i] < 50.0
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME - use CRSI extremes
            # LONG: HTF bull + CRSI oversold
            if htf_bull and crsi_oversold:
                if crsi[i] < 15.0:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: HTF bear + CRSI overbought
            elif htf_bear and crsi_overbought:
                if crsi[i] > 85.0:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        else:
            # TREND REGIME - use HMA + RSI
            # LONG: HTF bull + 4h HMA bull + RSI bull
            if htf_bull and hma_bull and rsi_bull:
                desired_signal = SIZE_BASE
            
            # SHORT: HTF bear + 4h HMA bear + RSI bear
            elif htf_bear and hma_bear and rsi_bear:
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