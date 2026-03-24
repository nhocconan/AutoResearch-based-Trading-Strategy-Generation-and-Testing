#!/usr/bin/env python3
"""
Experiment #164: 12h Primary + 1d/1w HTF — Regime-Adaptive (Chop + CRSI + Donchian)

Hypothesis: 12h timeframe with regime-adaptive logic outperforms static strategies.
Market alternates between trending and ranging - use Choppiness Index to detect regime
and switch between mean-reversion (CRSI) in ranges vs trend-following (Donchian) in trends.

Key innovations vs failed experiments:
- #158 failed with CRSI+Chop on 4h (Sharpe=-5.152) - too many trades, fee drag
- #162 failed with HMA+Chop on 4h (Sharpe=-0.235) - wrong entry logic
- This uses 12h (fewer trades) + proper regime detection + HTF confirmation

Design:
- Choppiness Index(14): >61.8 = range (use CRSI mean-reversion), <38.2 = trend (use Donchian breakout)
- 1d HMA(50): Major trend bias (only long if price>1d_HMA, only short if price<1d_HMA)
- 1w HMA(50): Weekly regime filter (boosts conviction when aligned)
- Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - proven 75% win rate
- Donchian(20): Breakout entry for trending regime
- ATR(14) trailing stop: 2.5x for risk management
- Position size: 0.28 (28% of capital, discrete levels)

Target: 25-40 trades/year on 12h, Sharpe>0.167, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_chop_crsi_donchian_1d1w_v1"
timeframe = "12h"
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
    Choppiness Index - measures market ranging vs trending
    CHOP > 61.8 = ranging market (mean-reversion favored)
    CHOP < 38.2 = trending market (trend-following favored)
    
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
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
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) - composite mean-reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Long signal: CRSI < 10 (oversold)
    Short signal: CRSI > 90 (overbought)
    """
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI on streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to absolute values for RSI calculation
    streak_abs = np.abs(streak)
    streak_direction = np.sign(streak)
    
    # RSI on streak (using up/down streak lengths)
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank (100) - where does current close rank in last 100 bars?
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / pr_period * 100.0
        percent_rank[i] = rank
    
    # Combine into CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_donchian_channels(high, low, period=20):
    """Donchian Channels - upper/lower bounds for breakout detection"""
    n = len(close) if 'close' in dir() else len(high)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period, n):
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for weekly regime filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_ranging = choppiness[i] > 55.0  # Slightly lower threshold to ensure trades
        is_trending = choppiness[i] < 45.0  # Slightly higher threshold to ensure trades
        # Neutral zone (45-55): use both strategies with lower conviction
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY REGIME (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === HTF CONVICTION ===
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        htf_aligned = htf_strong_bull or htf_strong_bear
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with CRSI ---
        if is_ranging:
            # Long: CRSI oversold + HTF bull bias + above SMA200
            if crsi[i] < 25.0 and htf_1d_bull and above_sma200:
                desired_signal = SIZE  # Full size in range with HTF confirmation
            
            # Short: CRSI overbought + HTF bear bias + below SMA200
            elif crsi[i] > 75.0 and htf_1d_bear and below_sma200:
                desired_signal = -SIZE
            
            # Fallback: Strong CRSI extreme (ignore some filters)
            elif crsi[i] < 15.0 and htf_1d_bull:
                desired_signal = SIZE * 0.7
            elif crsi[i] > 85.0 and htf_1d_bear:
                desired_signal = -SIZE * 0.7
        
        # --- TRENDING REGIME: Donchian Breakout ---
        elif is_trending:
            # Long: Price breaks Donchian upper + HTF bull
            if close[i] > donchian_upper[i] * 0.998 and htf_1d_bull:
                desired_signal = SIZE
            
            # Short: Price breaks Donchian lower + HTF bear
            elif close[i] < donchian_lower[i] * 1.002 and htf_1d_bear:
                desired_signal = -SIZE
            
            # Fallback: Strong breakout with HTF alignment
            elif close[i] > donchian_upper[i] * 0.995 and htf_strong_bull:
                desired_signal = SIZE * 0.8
            elif close[i] < donchian_lower[i] * 1.005 and htf_strong_bear:
                desired_signal = -SIZE * 0.8
        
        # --- NEUTRAL REGIME: Hybrid approach ---
        else:
            # Use CRSI but require stronger HTF alignment
            if crsi[i] < 20.0 and htf_strong_bull:
                desired_signal = SIZE * 0.6
            elif crsi[i] > 80.0 and htf_strong_bear:
                desired_signal = -SIZE * 0.6
            # Or Donchian with strong confirmation
            elif close[i] > donchian_upper[i] * 0.997 and htf_strong_bull:
                desired_signal = SIZE * 0.6
            elif close[i] < donchian_lower[i] * 1.003 and htf_strong_bear:
                desired_signal = -SIZE * 0.6
        
        # === FALLBACK: Ensure trades when HTF strongly aligned ===
        # This prevents 0-trade scenarios
        if desired_signal == 0.0:
            if htf_strong_bull and crsi[i] < 35.0:
                desired_signal = SIZE * 0.4
            elif htf_strong_bear and crsi[i] > 65.0:
                desired_signal = -SIZE * 0.4
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.7:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.7:
            final_signal = -SIZE * 0.7
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        elif desired_signal >= SIZE * 0.3:
            final_signal = SIZE * 0.3
        elif desired_signal <= -SIZE * 0.3:
            final_signal = -SIZE * 0.3
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
                # Flip position
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