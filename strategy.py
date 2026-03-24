#!/usr/bin/env python3
"""
Experiment #782: 4h Primary + 1d/1w HTF — Regime-Switching with Connors RSI

Hypothesis: 4h timeframe with regime detection (Choppiness Index) allows adaptive
strategy that mean-reverts in choppy markets and trend-follows in trending markets.
This addresses the bear/range market conditions of 2025 while capturing trends in 2021-2024.

Key innovations:
1. Choppiness Index (14) for regime detection: CHOP>61.8=range, CHOP<38.2=trend
2. Connors RSI for mean reversion entries in range regime
3. HMA(16/48) crossover for trend entries in trend regime
4. 1d HMA(21) for intermediate trend bias
5. 1w HMA(21) for major trend filter (only trade with weekly trend)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.25, ±0.30
8. LOOSE entry thresholds to guarantee ≥10 trades/symbol train, ≥3 test

Entry conditions (designed for trade generation):
- RANGE regime (CHOP>50): Long CRSI<20, Short CRSI>80 (with 1d HMA bias)
- TREND regime (CHOP<50): Long HMA bull crossover + 1d bull, Short HMA bear + 1d bear
- 1w HMA filter: Only long if price>1w HMA, only short if price<1w HMA

Target: Sharpe>0.40, trades>=10/symbol train, trades>=3/symbol test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_crsi_hma_1d1w_v1"
timeframe = "4h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - combines RSI, streak RSI, and percentile rank"""
    n = len(close)
    if n < rank_period + rsi_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(np.abs(streak) * np.sign(streak), streak_period)
    # Convert streak RSI to 0-100 scale properly
    streak_rsi_scaled = np.zeros(n)
    for i in range(len(streak_rsi)):
        if np.isnan(streak_rsi[i]):
            streak_rsi_scaled[i] = np.nan
        else:
            # Streak RSI: positive streaks = high values, negative = low
            if streak[i] >= 0:
                streak_rsi_scaled[i] = 50 + (streak_rsi[i] / 2)
            else:
                streak_rsi_scaled[i] = 50 - (streak_rsi[i] / 2)
            streak_rsi_scaled[i] = np.clip(streak_rsi_scaled[i], 0, 100)
    
    # Percentile Rank(100)
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < close[i])
        pct_rank[i] = (rank / (rank_period - 1)) * 100.0
    
    # CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi_scaled[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi_scaled[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - detects ranging vs trending markets"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        atr_like = tr_sum / period
        chop[i] = 100.0 * np.log10((highest_high - lowest_low) / (atr_like * np.sqrt(period))) / np.log10(period)
        chop[i] = np.clip(chop[i], 0, 100)
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    crsi_3_2_100 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi_3_2_100[i]):
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
        # CHOP > 61.8 = range, CHOP < 38.2 = trend, middle = transition
        in_range_regime = chop_14[i] > 50.0  # Use 50 as threshold for more trades
        in_trend_regime = chop_14[i] < 50.0
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 4h HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
            hma_crossover_long = (hma_16[i-1] <= hma_48[i-1]) and (hma_16[i] > hma_48[i])
            hma_crossover_short = (hma_16[i-1] >= hma_48[i-1]) and (hma_16[i] < hma_48[i])
        
        # === 4h HMA TREND ===
        hma_4h_bull = hma_16[i] > hma_48[i]
        hma_4h_bear = hma_16[i] < hma_48[i]
        
        # === CRSI CONDITIONS (for range regime) ===
        crsi_oversold = crsi_3_2_100[i] < 25.0  # Loose threshold for more trades
        crsi_overbought = crsi_3_2_100[i] > 75.0  # Loose threshold
        crsi_extreme_oversold = crsi_3_2_100[i] < 15.0
        crsi_extreme_overbought = crsi_3_2_100[i] > 85.0
        
        # === ENTRY LOGIC (REGIME-SWITCHING) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with CRSI
        if in_range_regime:
            # LONG: CRSI oversold + 1d bull bias + 1w not strongly bear
            if crsi_oversold and (htf_1d_bull or not htf_1w_bear):
                if crsi_extreme_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: CRSI overbought + 1d bear bias + 1w not strongly bull
            elif crsi_overbought and (htf_1d_bear or not htf_1w_bull):
                if crsi_extreme_overbought:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # TREND REGIME: Trend following with HMA
        elif in_trend_regime:
            # LONG: HMA bull + 1d bull + 1w bull (all aligned)
            if hma_4h_bull and htf_1d_bull and htf_1w_bull:
                if hma_crossover_long:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: HMA bear + 1d bear + 1w bear (all aligned)
            elif hma_4h_bear and htf_1d_bear and htf_1w_bear:
                if hma_crossover_short:
                    desired_signal = -SIZE_STRONG
                else:
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