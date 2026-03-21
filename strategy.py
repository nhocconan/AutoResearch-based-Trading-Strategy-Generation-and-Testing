#!/usr/bin/env python3
"""
Experiment #283: 15m Connors RSI Mean Reversion with 4h/1h HMA Trend Filter
Hypothesis: 15m timeframe is noisy but offers frequent mean reversion opportunities.
Connors RSI (CRSI) combines 3-period RSI + streak RSI + percentile rank for superior
entry timing (75% win rate in research). 4h HMA provides primary trend bias, 1h HMA
confirms intermediate direction. Choppiness Index filters regime - only mean revert
in choppy markets (CHOP>50), trend follow in trending markets (CHOP<50).
Loose entry thresholds ensure sufficient trades (>10 train, >3 test per symbol).
Position sizing: 0.25 entry, 0.125 half at 2R profit. Stoploss: 2.0*ATR trailing.
Target: Beat Sharpe=0.499 from current best while ensuring ALL symbols have Sharpe>0.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_chop_4h_1h_hma_mean_reversion_atr_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percentile Rank - where current return ranks vs last 100 returns
    returns = np.diff(close, prepend=close[0]) / np.where(close > 0, close, 1)
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = np.sum(window < current) / len(window) * 100
    
    # CRSI
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range > 0, price_range, 1e-10)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    ma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    std = np.where(std > 0, std, 1e-10)
    zscore = (close - ma) / std
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    
    # Track previous values
    prev_crsi = np.roll(crsi, 1)
    prev_crsi[0] = crsi[0]
    prev_zscore = np.roll(zscore, 1)
    prev_zscore[0] = zscore[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        hma_1h_bullish = close[i] > hma_1h_aligned[i]
        hma_1h_bearish = close[i] < hma_1h_aligned[i]
        
        # Regime detection
        choppy_regime = chop[i] > 50  # Range market - favor mean reversion
        trending_regime = chop[i] < 50  # Trend market - favor trend following
        
        # CRSI signals (loose thresholds for more trades)
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_rising = crsi[i] > prev_crsi[i] and crsi[i] < 40
        crsi_falling = crsi[i] < prev_crsi[i] and crsi[i] > 60
        
        # Z-score signals
        zscore_extreme_low = zscore[i] < -1.5
        zscore_extreme_high = zscore[i] > 1.5
        zscore_mean_cross_up = prev_zscore[i] < 0 and zscore[i] > -0.5
        zscore_mean_cross_down = prev_zscore[i] > 0 and zscore[i] < 0.5
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Mean reversion in choppy regime (primary signal)
        if choppy_regime:
            if crsi_oversold and zscore_extreme_low:
                new_signal = SIZE_ENTRY
            elif crsi_rising and zscore_mean_cross_up:
                new_signal = SIZE_ENTRY
            elif crsi[i] < 30 and hma_4h_bullish:
                new_signal = SIZE_ENTRY
        
        # Trend following in trending regime
        elif trending_regime:
            if hma_4h_bullish and hma_1h_bullish:
                if crsi[i] < 40 and crsi_rising:
                    new_signal = SIZE_ENTRY
                elif zscore_mean_cross_up and crsi[i] < 50:
                    new_signal = SIZE_ENTRY
        
        # Strong bullish confirmation (both HTF aligned)
        if hma_4h_bullish and hma_1h_bullish:
            if crsi[i] < 35 and zscore[i] < -1.0:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Mean reversion in choppy regime (primary signal)
        if choppy_regime:
            if crsi_overbought and zscore_extreme_high:
                new_signal = -SIZE_ENTRY
            elif crsi_falling and zscore_mean_cross_down:
                new_signal = -SIZE_ENTRY
            elif crsi[i] > 70 and hma_4h_bearish:
                new_signal = -SIZE_ENTRY
        
        # Trend following in trending regime
        elif trending_regime:
            if hma_4h_bearish and hma_1h_bearish:
                if crsi[i] > 60 and crsi_falling:
                    new_signal = -SIZE_ENTRY
                elif zscore_mean_cross_down and crsi[i] > 50:
                    new_signal = -SIZE_ENTRY
        
        # Strong bearish confirmation (both HTF aligned)
        if hma_4h_bearish and hma_1h_bearish:
            if crsi[i] > 65 and zscore[i] > 1.0:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals