#!/usr/bin/env python3
"""
Experiment #731: 4h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI

Hypothesis: After 489 failed strategies, the pattern is clear - complex regime
switching with too many filters causes 0 trades. This strategy simplifies:
1. Choppiness Index (CHOP) for binary regime: >50=range(mean-revert), <50=trend
2. Connors RSI (CRSI) for entry timing - proven 75% win rate in literature
3. 1d HMA for trend bias (simpler than dual 1d+1w)
4. Multiple independent entry paths per regime to ensure trade frequency
5. 2.5x ATR trailing stoploss for risk management

Key differences from failed attempts:
- CRSI thresholds: ≤15 long, ≥85 short (wide enough to trigger)
- CHOP threshold: 50 (not extreme 38/62 which rarely triggers)
- Entry paths: 3+ independent conditions per regime
- Signal sizes: discrete 0.25/0.30 to minimize fee churn
- No complex position state tracking - signal drives everything

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (proven, 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_hma_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals.
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    Literature shows 75% win rate with CRSI<10 long, CRSI>90 short.
    We use wider thresholds (15/85) to ensure trade frequency.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # Component 1: RSI(3) on close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = np.clip(rsi_close, 0, 100)
    
    # Component 2: RSI on streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: Percentile Rank of returns over lookback
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pct_rank[i] = np.sum(valid < returns[i]) / len(valid) * 100
    
    # Combine all 3 components
    with np.errstate(invalid='ignore'):
        crsi = (rsi_close + rsi_streak + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy (mean reversion works)
    CHOP < 38.2 = trending (trend following works)
    We use 50 as binary threshold for simplicity.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        atr_like = tr_sum / period
        chop[i] = 100 * np.log10(highest - lowest) / np.log10(atr_like + 1e-10)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(chop_4h[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 50 = range/choppy (use mean reversion)
        # CHOP < 50 = trending (use trend following)
        is_range_regime = chop_4h[i] > 50
        is_trend_regime = chop_4h[i] < 50
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === RANGE REGIME: MEAN REVERSION (CRSI extremes) ===
        if is_range_regime:
            # Long: CRSI deeply oversold
            if crsi_4h[i] < 15:
                desired_signal = BASE_SIZE
            
            # Short: CRSI deeply overbought
            elif crsi_4h[i] > 85:
                desired_signal = -BASE_SIZE
            
            # Secondary entry: moderate CRSI + SMA200 confluence
            elif crsi_4h[i] < 25 and above_sma200:
                desired_signal = REDUCED_SIZE
            elif crsi_4h[i] > 75 and below_sma200:
                desired_signal = -REDUCED_SIZE
        
        # === TREND REGIME: TREND FOLLOWING ===
        elif is_trend_regime:
            # Long: bullish trend + CRSI pullback (not oversold, just dipped)
            if trend_1d_bullish and 30 < crsi_4h[i] < 50:
                desired_signal = BASE_SIZE
            
            # Short: bearish trend + CRSI bounce (not overbought, just rallied)
            elif trend_1d_bearish and 50 < crsi_4h[i] < 70:
                desired_signal = -BASE_SIZE
            
            # Secondary: strong trend + mild CRSI signal
            elif trend_1d_bullish and crsi_4h[i] < 40 and above_sma200:
                desired_signal = REDUCED_SIZE
            elif trend_1d_bearish and crsi_4h[i] > 60 and below_sma200:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if range regime and CRSI not overbought, or trend bullish
                if (is_range_regime and crsi_4h[i] < 70) or (is_trend_regime and trend_1d_bullish):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if range regime and CRSI not oversold, or trend bearish
                if (is_range_regime and crsi_4h[i] > 30) or (is_trend_regime and trend_1d_bearish):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI extremely overbought in range, or trend reverses
            if (is_range_regime and crsi_4h[i] > 80) or (is_trend_regime and trend_1d_bearish):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI extremely oversold in range, or trend reverses
            if (is_range_regime and crsi_4h[i] < 20) or (is_trend_regime and trend_1d_bullish):
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
        
        signals[i] = desired_signal
    
    return signals