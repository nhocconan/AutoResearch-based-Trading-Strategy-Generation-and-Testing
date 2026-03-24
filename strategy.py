#!/usr/bin/env python3
"""
Experiment #019: 4h Primary + 1d HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: After 18 experiments, the key insight is that single-regime strategies
fail because BTC/ETH behave differently in bull vs bear vs range markets. The 2025
test period is bear/range, which destroys pure trend-following strategies.

Solution: DUAL REGIME approach
1. Choppiness Index (CHOP) detects regime: CHOP>61.8=range, CHOP<38.2=trend
2. Range regime: Connors RSI mean reversion (75% win rate in research)
3. Trend regime: KAMA + ADX trend following
4. 1d HTF KAMA for overall bias filter

This should work in BOTH bull (trend) and bear/range (mean revert) markets.

Entry Logic:
- Range (CHOP>55): Long when CRSI<15 + price>SMA200, Short when CRSI>85 + price<SMA200
- Trend (CHOP<45): Long when close>KAMA + ADX>20 + RSI>40, Short when close<KAMA + ADX>20 + RSI<60
- 1d KAMA must agree with direction (bias filter)
- Size: 0.28 (discrete, minimizes fee churn)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_crsi_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    er = np.zeros(n)
    
    for i in range(slow_period, n):
        price_change = abs(close[i] - close[i - slow_period])
        if price_change < 1e-10:
            er[i] = 0.0
        else:
            volatility = np.sum(np.abs(np.diff(close[i - slow_period:i + 1])))
            er[i] = price_change / volatility if volatility > 1e-10 else 0.0
    
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = np.zeros(n)
    for i in range(period * 2, n):
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum > 1e-10 else 0.0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_rsi(close, period=14):
    """RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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
    """
    Choppiness Index - measures if market is trending or ranging
    CHOP > 61.8 = ranging (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR,n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j], 
                     abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10 or atr_sum < 1e-10:
            choppiness[i] = 100.0
        else:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - mean reversion indicator
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    Long when CRSI < 10-15, Short when CRSI > 85-90
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (convert to positive for RSI calc)
    streak_for_rsi = np.abs(streak)
    streak_rsi = calculate_rsi(streak_for_rsi, streak_period)
    
    # Percent Rank - where does current close rank in last 100 bars?
    for i in range(rank_period, n):
        window = close[i - rank_period:i]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        
        if np.isnan(rsi3[i]) or np.isnan(streak_rsi[i]):
            crsi[i] = np.nan
        else:
            crsi[i] = (rsi3[i] + streak_rsi[i] + rank) / 3.0
    
    return crsi

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    cumsum = np.cumsum(close)
    sma[period-1:] = (cumsum[period-1:] - np.concatenate([[0], cumsum[:-period]])) / period
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for HTF trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=21)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_ranging = choppiness[i] > 55.0  # Range regime
        is_trending = choppiness[i] < 45.0  # Trend regime
        # 45-55 is transition zone - stay flat
        
        # === HTF BIAS (1d KAMA) ===
        htf_bullish = close[i] > kama_1d_aligned[i]
        htf_bearish = close[i] < kama_1d_aligned[i]
        
        # === SMA200 FILTER (for mean reversion) ===
        above_sma200 = not np.isnan(sma200[i]) and close[i] > sma200[i]
        below_sma200 = not np.isnan(sma200[i]) and close[i] < sma200[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_ranging and not np.isnan(crsi[i]):
            # RANGE REGIME: Connors RSI mean reversion
            # Long: CRSI oversold + above SMA200 + HTF bullish bias
            if crsi[i] < 15.0 and above_sma200 and htf_bullish:
                desired_signal = SIZE
            # Short: CRSI overbought + below SMA200 + HTF bearish bias
            elif crsi[i] > 85.0 and below_sma200 and htf_bearish:
                desired_signal = -SIZE
        
        elif is_trending:
            # TREND REGIME: KAMA + ADX trend following
            kama_4h_bull = close[i] > kama_4h[i]
            kama_4h_bear = close[i] < kama_4h[i]
            trend_strong = adx[i] > 20.0
            
            # Long: 4h KAMA bull + 1d KAMA bull + ADX strong + RSI not overbought
            if kama_4h_bull and htf_bullish and trend_strong and rsi[i] > 40.0 and rsi[i] < 75.0:
                desired_signal = SIZE
            # Short: 4h KAMA bear + 1d KAMA bear + ADX strong + RSI not oversold
            elif kama_4h_bear and htf_bearish and trend_strong and rsi[i] < 60.0 and rsi[i] > 25.0:
                desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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