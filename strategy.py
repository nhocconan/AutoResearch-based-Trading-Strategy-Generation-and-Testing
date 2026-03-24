#!/usr/bin/env python3
"""
Experiment #077: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI + Donchian)

Hypothesis: Daily timeframe with weekly trend bias will capture major moves while
avoiding whipsaw. Using Choppiness Index to switch between mean-reversion (chop)
and trend-following (breakout) regimes. Connors RSI for MR entries, Donchian for
trend entries. This should work across BTC/ETH/SOL in both bull and bear markets.

Key design:
1. 1w HMA for long-term trend bias (only trade with HTF trend)
2. 1d Choppiness Index regime: CHOP>55 = range (MR), CHOP<45 = trend (breakout)
3. Connors RSI<20 for long MR, >80 for short MR (looser than typical)
4. Donchian(20) breakout with volume confirmation for trend entries
5. ATR(14) 3x trailing stoploss
6. Size: 0.30 trend, 0.25 MR (discrete levels to minimize churn)

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_dual_regime_crsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hull_raw = 2.0 * wma_half - wma_full
    hma = pd.Series(hull_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Formula: 100 * (SUM(ATR, n) / (Highest High - Lowest Low)) / (log10(n))
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low and sum_atr > 0:
            chop[i] = 100.0 * (sum_atr / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return np.clip(chop, 0, 100)

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator
    """
    n = len(close)
    if n < pr_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = streak[i - 1]
    
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        window = streak[max(0, i-streak_period):i+1]
        max_abs = np.max(np.abs(window))
        if max_abs > 0:
            streak_rsi[i] = 50.0 + (streak[i] / max_abs) * 50.0
        else:
            streak_rsi[i] = 50.0
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank
    pr = np.zeros(n)
    for i in range(pr_period, n):
        returns = np.diff(close[max(0, i-pr_period):i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            pr[i] = 100.0 * np.sum(returns < current_return) / len(returns)
    
    # Combine
    for i in range(pr_period, n):
        crsi[i] = (rsi_short[i] + streak_rsi[i] + pr[i]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for HTF trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_MR = 0.25
    
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
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
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
        if np.isnan(sma_200[i]) or np.isnan(hma_1w_aligned[i]):
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
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_choppy = chop[i] > 50.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === SMA200 TREND FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === HMA CROSSOVER ===
        hma_bull = hma_21[i] > hma_50[i]
        hma_bear = hma_21[i] < hma_50[i]
        
        # === CONNORS RSI MEAN REVERSION ===
        crsi_oversold = crsi[i] < 25.0  # Looser threshold for more trades
        crsi_overbought = crsi[i] > 75.0  # Looser threshold
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above prev high
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below prev low
        vol_confirmed = volume[i] > 1.2 * vol_ma[i]  # 20% above avg volume
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TREND FOLLOWING: HTF + HMA + Donchian breakout + volume
        if is_trending or not is_choppy:
            if htf_bull and hma_bull and above_sma200 and donchian_breakout_long and vol_confirmed:
                desired_signal = SIZE_TREND
            elif htf_bear and hma_bear and below_sma200 and donchian_breakout_short and vol_confirmed:
                desired_signal = -SIZE_TREND
        
        # MEAN REVERSION: CRSI extremes in choppy regime with HTF bias
        if is_choppy:
            if htf_bull and crsi_oversold and above_sma200:
                desired_signal = SIZE_MR
            elif htf_bear and crsi_overbought and below_sma200:
                desired_signal = -SIZE_MR
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
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