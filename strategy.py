#!/usr/bin/env python3
"""
Experiment #466: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime

Hypothesis: Daily timeframe with weekly bias should capture multi-week trends
while avoiding the noise of lower TFs. Based on research showing:
- Connors RSI (CRSI) has 75% win rate for mean reversion
- Choppiness Index is the BEST regime filter (better than ADX)
- ETH achieved Sharpe +0.923 with CRSI + Choppiness on 1d

New approach:
1. CONNORS RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI for reversal detection
2. CHOPPINESS INDEX: CHOP(14) > 61.8 = range, < 38.2 = trend
   - cleaner regime separation than ADX
3. WEEKLY HMA BIAS: Only take signals aligned with 1w trend
   - Reduces counter-trend trades that fail in 2022-style crashes
4. LOOSE ENTRY THRESHOLDS: CRSI < 20 / > 80 (not extreme 10/90)
   - Ensures we get 30-60 trades/year on 1d (Rule 9: MUST generate trades)
5. ATR STOPLOSS: 2.5x ATR from entry, signal → 0 when hit

Entry Logic:
- Range Long: CHOP > 61.8 + CRSI < 20 + price > SMA200 + 1w HMA bull
- Range Short: CHOP > 61.8 + CRSI > 80 + price < SMA200 + 1w HMA bear
- Trend Long: CHOP < 38.2 + HMA cross + 1w HMA bull + Donchian breakout
- Trend Short: CHOP < 38.2 + HMA cross + 1w HMA bear + Donchian breakdown

Target: Sharpe>0.45, DD>-35%, trades>=40 train (10/year), trades>=6 test
Timeframe: 1d (proven to work better than lower TFs for BTC/ETH)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_v1"
timeframe = "1d"
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

def calculate_rsi_streak(close, period=2):
    """RSI of consecutive up/down days (Connors RSI component)"""
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Calculate streak: consecutive up or down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if streak[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        window = streak[max(0, i-period):i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            avg_streak = np.mean(valid)
            # Map streak to 0-100 scale
            # streak of +period = 100, -period = 0
            streak_rsi[i] = 50.0 + (avg_streak / period) * 50.0
            streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    return streak_rsi

def calculate_percentile_rank(values, period=100):
    """Percentile rank for Connors RSI component"""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = values[max(0, i-period+1):i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pr[i] = np.sum(valid < values[i]) / len(valid) * 100.0
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    
    rsi_fast = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percentile_rank(close, pr_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(max(rsi_period, streak_period, pr_period), n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_fast[i] + rsi_streak[i] + pr[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and sum_atr > 1e-10:
            choppiness[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return choppiness

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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for weekly trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_8 = calculate_hma(close, period=8)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range/choppy (mean reversion)
        # CHOP < 38.2 = trending (breakout)
        # Between = maintain previous regime
        
        is_choppy = choppiness[i] > 61.8
        is_trending = choppiness[i] < 38.2
        
        # === WEEKLY HMA BIAS ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES (LOOSE: 20/80 not 10/90) ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === HMA CROSSOVER (8/21) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_8[i]) and not np.isnan(hma_8[i-1]):
            if not np.isnan(hma_21[i]) and not np.isnan(hma_21[i-1]):
                if hma_8[i-1] <= hma_21[i-1] and hma_8[i] > hma_21[i]:
                    hma_cross_long = True
                if hma_8[i-1] >= hma_21[i-1] and hma_8[i] < hma_21[i]:
                    hma_cross_short = True
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ENTRY LOGIC (LOOSE - ensure we get trades) ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (Connors RSI mean reversion)
        if is_choppy:
            # Long: CRSI < 20 + above SMA200 + weekly bull bias
            if crsi_oversold and above_sma200 and htf_bull:
                desired_signal = SIZE_BASE
            
            # Short: CRSI > 80 + below SMA200 + weekly bear bias
            elif crsi_overbought and below_sma200 and htf_bear:
                desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (HMA + Donchian breakout)
        elif is_trending:
            # Long: Weekly bull + (HMA cross OR Donchian breakout)
            if htf_bull:
                if hma_cross_long or donchian_breakout_long:
                    desired_signal = SIZE_STRONG
            
            # Short: Weekly bear + (HMA cross OR Donchian breakdown)
            elif htf_bear:
                if hma_cross_short or donchian_breakdown_short:
                    desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals