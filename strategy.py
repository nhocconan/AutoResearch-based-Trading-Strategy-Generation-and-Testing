#!/usr/bin/env python3
"""
Experiment #068: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: Combining Choppiness Index regime detection with Connors RSI entries and UTC 
session filtering will generate 30-80 trades/year with positive Sharpe across all symbols.

Key innovations:
1. CHOP > 55 = range mode (mean revert with CRSI extremes)
2. CHOP < 45 = trend mode (follow 4h HMA direction)
3. Session filter: only trade 8-20 UTC (highest volume hours)
4. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
5. Loose CRSI thresholds (15/85 not 10/90) to ensure trades generate
6. 4h/1d HMA as soft bias, not hard filter (allows trades in both directions)
7. ATR 2.5x trailing stop to limit drawdown
8. Volume confirmation > 0.7x avg (not 1.0x to allow more trades)

Target: Sharpe>0.35, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 30m (target 30-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - less lag than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    
    def wma(data, span):
        res = np.full(len(data), np.nan)
        w = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            res[i] = np.sum(data[i - span + 1:i + 1] * w) / np.sum(w)
        return res
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    double_wma = 2.0 * wma_half - wma_full
    hma = wma(double_wma, sqrt_p)
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    Formula: 100 * (SUM(ATR, n) / (Highest High - Lowest Low)) / (log10(n))
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    chop = np.full(n, np.nan)
    for i in range(period, n):
        range_hl = hh[i] - ll[i]
        if range_hl > 1e-10:
            chop[i] = 100.0 * (atr_sum[i] / range_hl) / np.log10(period)
        else:
            chop[i] = 100.0
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    if n < pr_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        # Calculate RSI of streak values
        streak_gain = np.where(streak[:i+1] > 0, streak[:i+1], 0.0)
        streak_loss = np.where(streak[:i+1] < 0, -streak[:i+1], 0.0)
        if len(streak_gain) >= streak_period:
            avg_sg = np.mean(streak_gain[-streak_period:])
            avg_sl = np.mean(streak_loss[-streak_period:])
            if avg_sl < 1e-10:
                streak_rsi[i] = 100.0
            else:
                rs = avg_sg / avg_sl
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100) - where current close ranks in last 100 bars
    pr = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        rank = np.sum(window < close[i])
        pr[i] = 100.0 * rank / (pr_period - 1)
    
    # Connors RSI
    crsi = (rsi + streak_rsi + pr) / 3.0
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation filter"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    return pd.to_datetime(open_time, unit='ms').hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h/1d HMA for HTF trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # Fast HMA for 30m trend confirmation
    hma_fast = calculate_hma(close, period=8)
    hma_slow = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30  # Larger size for trend trades
    SIZE_MR = 0.20     # Smaller size for mean reversion
    
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
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === HTF BIAS (4h and 1d HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # HTF alignment score (-2 to +2)
        htf_score = 0
        if hma_4h_bull: htf_score += 1
        if hma_1d_bull: htf_score += 1
        if hma_4h_bear: htf_score -= 1
        if hma_1d_bear: htf_score -= 1
        
        # === REGIME (Choppiness Index) ===
        is_ranging = chop[i] > 55.0  # Choppy/range market
        is_trending = chop[i] < 45.0  # Trending market
        # Neutral zone 45-55: use either logic
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0  # Loose threshold for trade generation
        crsi_overbought = crsi[i] > 80.0  # Loose threshold for trade generation
        
        # === 30m TREND (HMA crossover) ===
        hma_cross_bull = hma_fast[i] > hma_slow[i]
        hma_cross_bear = hma_fast[i] < hma_slow[i]
        
        # === VOLUME CONFIRMATION ===
        vol_above_avg = volume[i] > 0.7 * vol_sma[i]  # Loose volume filter
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if in_session:
            # MEAN REVERSION PATH (CHOP > 55, ranging market)
            if is_ranging:
                # Long: CRSI oversold + volume + HTF not strongly bearish
                if crsi_oversold and vol_above_avg and htf_score >= -1:
                    desired_signal = SIZE_MR
                # Short: CRSI overbought + volume + HTF not strongly bullish
                elif crsi_overbought and vol_above_avg and htf_score <= 1:
                    desired_signal = -SIZE_MR
            
            # TREND FOLLOWING PATH (CHOP < 45, trending market)
            elif is_trending:
                # Long: HTF bullish + 30m HMA bull + volume
                if htf_score >= 1 and hma_cross_bull and vol_above_avg:
                    desired_signal = SIZE_TREND
                # Short: HTF bearish + 30m HMA bear + volume
                elif htf_score <= -1 and hma_cross_bear and vol_above_avg:
                    desired_signal = -SIZE_TREND
            
            # NEUTRAL ZONE (45 <= CHOP <= 55)
            else:
                # Use CRSI mean reversion with HTF bias
                if crsi_oversold and htf_score >= 0 and vol_above_avg:
                    desired_signal = SIZE_MR
                elif crsi_overbought and htf_score <= 0 and vol_above_avg:
                    desired_signal = -SIZE_MR
        
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
        if desired_signal >= SIZE_TREND * 0.85:
            final_signal = SIZE_TREND
        elif desired_signal >= SIZE_MR * 0.85:
            final_signal = SIZE_MR
        elif desired_signal <= -SIZE_TREND * 0.85:
            final_signal = -SIZE_TREND
        elif desired_signal <= -SIZE_MR * 0.85:
            final_signal = -SIZE_MR
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