#!/usr/bin/env python3
"""
Experiment #279: 1h Primary + 4h/12h HTF — Asymmetric Regime + cRSI + Session Filter

Hypothesis: 1h timeframe with strict 3+ confluence filters can achieve 40-80 trades/year
while maintaining positive Sharpe across all symbols. Key innovations:

1. ASYMMETRIC REGIME: In bear markets (2025+), favor short signals; in bull, favor long
2. SESSION FILTER: Only trade 08-20 UTC (high liquidity, less noise)
3. DUAL HTF CONFIRMATION: 4h HMA for intermediate, 12h HMA for major trend
4. CHOPPY vs TRENDING: Different logic per regime (mean revert vs breakout)
5. CONNORS RSI: More reliable than standard RSI for extremes (75% win rate in research)

Regime Detection:
- Choppiness(14) > 60 = choppy → Connors RSI mean reversion
- Choppiness(14) < 45 = trending → Donchian breakout + HTF trend

Entry Confluence (ALL required):
- LONG: 4h HMA bullish OR 12h HMA bullish + CRSI < 15 + Session 08-20 UTC
- SHORT: 4h HMA bearish OR 12h HMA bearish + CRSI > 85 + Session 08-20 UTC

Position sizing: 0.20 base, 0.30 strong (discrete levels to minimize churn)
Stoploss: 2.5x ATR trailing

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_asymmetric_regime_crsi_session_4h12h_v1"
timeframe = "1h"
leverage = 1.0

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
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
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < pr_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Streak RSI
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        up_count = np.sum(streak[max(0, i-streak_period+1):i+1] > 0)
        streak_rsi[i] = 100.0 * up_count / streak_period
    
    # Percent Rank of returns
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(pr_period, n):
        window = returns[i-pr_period:i]
        if len(window) > 0 and not np.isnan(returns[i]):
            count_below = np.sum(window < returns[i])
            percent_rank[i] = 100.0 * count_below / len(window)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
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
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        hour = extract_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === REGIME DETECTION ===
        choppy_threshold = 60.0
        trending_threshold = 45.0
        
        is_choppy = chop[i] > choppy_threshold
        is_trending = chop[i] < trending_threshold
        
        # === HTF BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Combined HTF bias (at least one confirms)
        htf_bullish = htf_4h_bull or htf_12h_bull
        htf_bearish = htf_4h_bear or htf_12h_bear
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # CHOPPY REGIME: Mean reversion with CRSI
        if is_choppy and in_session:
            # Long: oversold + HTF not strongly bearish + above SMA200
            if crsi_oversold and (htf_bullish or above_sma200):
                desired_signal = SIZE_BASE
            
            # Short: overbought + HTF not strongly bullish + below SMA200
            elif crsi_overbought and (htf_bearish or below_sma200):
                desired_signal = -SIZE_BASE
        
        # TRENDING REGIME: Breakout with HTF confirmation
        elif is_trending and in_session:
            # Long: breakout + HTF bullish
            if breakout_long and htf_bullish:
                desired_signal = SIZE_STRONG
            
            # Short: breakout + HTF bearish
            elif breakout_short and htf_bearish:
                desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
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
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
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