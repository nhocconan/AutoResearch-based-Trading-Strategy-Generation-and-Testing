#!/usr/bin/env python3
"""
Experiment #1331: 4h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Market regime detection via Choppiness Index allows switching between
trend-following (low CHOP) and mean-reversion (high CHOP). Combined with Connors RSI
for faster entry signals and 1d HMA for macro trend filter.

Key design:
1. Choppiness Index (14) regime filter: CHOP>61.8=range, CHOP<38.2=trend
2. Connors RSI (3,2,100) for faster mean-reversion signals than standard RSI
3. 1d HMA(21) for macro trend bias via mtf_data alignment
4. 4h HMA(16) for local trend confirmation
5. Regime-adaptive entries:
   - Trend: Pullback to HMA + CRSI 30-50 (long) or 50-70 (short)
   - Range: CRSI<10 long, CRSI>90 short
6. ATR(14) trailing stop at 2.5x for risk management
7. Size: 0.30 discrete levels

Target: 25-50 trades/year on 4h, Sharpe > 0.612, trades >= 30 train, >= 5 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
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
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            
            if tr_sum > 0:
                chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Designed for short-term mean reversion with 75%+ win rate
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3) - very fast
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_rsi3 = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_rsi3 = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi3 = np.full(n, np.nan)
    mask = loss_rsi3 > 1e-10
    rsi3[mask] = 100.0 - (100.0 / (1.0 + gain_rsi3[mask] / loss_rsi3[mask]))
    rsi3[loss_rsi3 <= 1e-10] = 100.0
    rsi3[:rsi_period] = np.nan
    
    # RSI Streak (2) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask2 = streak_loss_smooth > 1e-10
    rsi_streak[mask2] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask2] / streak_loss_smooth[mask2]))
    rsi_streak[streak_loss_smooth <= 1e-10] = 100.0
    rsi_streak[:streak_period+5] = np.nan
    
    # Percent Rank (100) - where current return ranks vs last 100
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    pct_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if not np.any(np.isnan(window)):
            current = returns[i]
            count_below = np.sum(window < current)
            pct_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi3[i] + rsi_streak[i] + pct_rank[i]) / 3.0
    
    return crsi

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

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_16 = calculate_hma(close, period=16)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # HMA slope for trend direction
    hma_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(hma_16[i]) and not np.isnan(hma_16[i-1]):
            hma_slope[i] = hma_16[i] - hma_16[i-1]
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_slope[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop[i] < 45.0  # Trending market
        is_range_regime = chop[i] > 55.0  # Range/choppy market
        # Neutral zone 45-55: use trend bias
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === LOCAL TREND (4h HMA) ===
        hma_bull = (close[i] > hma_16[i]) and (hma_slope[i] > 0)
        hma_bear = (close[i] < hma_16[i]) and (hma_slope[i] < 0)
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow macro trend with pullback entries
        if is_trend_regime:
            # LONG: Macro bull + pullback CRSI
            if macro_bull and hma_bull:
                # CRSI pullback in uptrend (30-50 range)
                if 25.0 <= crsi[i] <= 50.0:
                    desired_signal = BASE_SIZE
                # CRSI breaking above 35 with momentum
                elif 35.0 < crsi[i] < 55.0 and above_sma200:
                    desired_signal = BASE_SIZE
                # Deep pullback with SMA200 support
                elif crsi[i] < 30.0 and above_sma200:
                    desired_signal = BASE_SIZE
            
            # SHORT: Macro bear + bounce CRSI
            elif macro_bear and hma_bear:
                # CRSI bounce in downtrend (50-75 range)
                if 50.0 <= crsi[i] <= 75.0:
                    desired_signal = -BASE_SIZE
                # CRSI breaking below 65 with momentum
                elif 45.0 < crsi[i] < 65.0 and below_sma200:
                    desired_signal = -BASE_SIZE
                # Overbought rejection with SMA200 resistance
                elif crsi[i] > 70.0 and below_sma200:
                    desired_signal = -BASE_SIZE
        
        # RANGE REGIME: Mean revert at CRSI extremes
        elif is_range_regime:
            # LONG: CRSI oversold extreme
            if crsi[i] < 15.0:
                # Extra confirmation: near SMA200 or HMA support
                if above_sma200 or (close[i] < hma_16[i] * 0.98):
                    desired_signal = BASE_SIZE
            # SHORT: CRSI overbought extreme
            elif crsi[i] > 85.0:
                # Extra confirmation: near SMA200 or HMA resistance
                if below_sma200 or (close[i] > hma_16[i] * 1.02):
                    desired_signal = -BASE_SIZE
        
        # NEUTRAL ZONE: Use macro bias only at extremes
        else:
            if macro_bull and crsi[i] < 20.0:
                desired_signal = BASE_SIZE
            elif macro_bear and crsi[i] > 80.0:
                desired_signal = -BASE_SIZE
        
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
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
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