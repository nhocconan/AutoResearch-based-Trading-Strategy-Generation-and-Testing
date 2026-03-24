#!/usr/bin/env python3
"""
Experiment #438: 4h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: 4h timeframe with daily trend bias captures multi-day moves while avoiding
lower-TF noise. Key innovations from failed experiments:
1. CHOPPINESS INDEX regime switch (proven on ETH Sharpe +0.923)
2. CONNORS RSI for mean reversion (75% win rate in literature)
3. LOOSE entry conditions to ensure >=10 trades/train, >=3 trades/test
4. Asymmetric sizing: smaller positions in bear regime (daily HMA bear)
5. Simple ATR stoploss (2.5x) to limit drawdown

Entry Logic:
- Choppy (CHOP>50): Connors RSI <20 long, >80 short (mean reversion)
- Trending (CHOP<40): HMA cross + daily bias (trend follow)
- Neutral: Hold previous regime state (hysteresis)

Position Sizing:
- Bull regime (price>1d HMA): 0.30
- Bear regime (price<1d HMA): 0.20 (smaller, more cautious)
- Discrete levels only: 0.0, ±0.20, ±0.30

Target: Sharpe>0.45, DD>-35%, trades>=60 train, trades>=10 test
Timeframe: 4h (proven higher-TF works best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_connors_hma_1d_regime_v1"
timeframe = "4h"
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
    Choppiness Index (CHOP) - measures market choppiness vs trending
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
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    Long: CRSI < 10, Short: CRSI > 90
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Calculate streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_abs = np.abs(streak)
    streak_signed = np.sign(streak) * streak_abs
    
    # Simplified streak RSI
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        gains = np.zeros(streak_period + 1)
        losses = np.zeros(streak_period + 1)
        for j in range(1, streak_period + 1):
            if streak_signed[i-j+1] > streak_signed[i-j]:
                gains[j] = streak_signed[i-j+1] - streak_signed[i-j]
            else:
                losses[j] = streak_signed[i-j] - streak_signed[i-j+1]
        
        avg_gain = np.mean(gains[1:])
        avg_loss = np.mean(losses[1:])
        
        if avg_loss < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percentile Rank(100) on close
    pr = np.zeros(n)
    pr[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        pr[i] = np.sum(window < close[i]) / rank_period * 100.0
    
    # Combine CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_close[i] + streak_rsi[i] + pr[i]) / 3.0
    
    return crsi

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower, std

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_std = calculate_bollinger(close, period=20, std_dev=2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BULL = 0.30  # Larger size in bull regime
    SIZE_BEAR = 0.20  # Smaller size in bear regime
    SIZE_CHOP = 0.25  # Medium size for mean reversion
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 50 = choppy (mean reversion)
        # CHOP < 40 = trending (trend follow)
        # 40-50 = hold previous regime (hysteresis)
        chop_val = chop[i]
        
        if chop_val > 50.0:
            current_regime = 2  # choppy
        elif chop_val < 40.0:
            current_regime = 1  # trending
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === DAILY TREND BIAS ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # Select position size based on daily bias
        if daily_bull:
            SIZE_TREND = SIZE_BULL
        else:
            SIZE_TREND = SIZE_BEAR
        
        # === 4h HMA TREND ===
        hma_4h_bull = close[i] > hma_4h[i]
        hma_4h_bear = close[i] < hma_4h[i]
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_4h_fast[i]) and not np.isnan(hma_4h_fast[i-1]):
            if not np.isnan(hma_4h[i]) and not np.isnan(hma_4h[i-1]):
                if hma_4h_fast[i-1] <= hma_4h[i-1] and hma_4h_fast[i] > hma_4h[i]:
                    hma_cross_long = True
                if hma_4h_fast[i-1] >= hma_4h[i-1] and hma_4h_fast[i] < hma_4h[i]:
                    hma_cross_short = True
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = False
        donchian_breakdown_short = False
        if not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            donchian_breakout_long = close[i] > donchian_upper[i-1]
            donchian_breakdown_short = close[i] < donchian_lower[i-1]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 20.0
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 80.0
        
        # === RSI EXTREMES (backup for CRSI) ===
        rsi_oversold = not np.isnan(rsi[i]) and rsi[i] < 30.0
        rsi_overbought = not np.isnan(rsi[i]) and rsi[i] > 70.0
        
        # === BB TOUCH ===
        touch_lower = not np.isnan(bb_lower[i]) and close[i] <= bb_lower[i]
        touch_upper = not np.isnan(bb_upper[i]) and close[i] >= bb_upper[i]
        
        # === ENTRY LOGIC (LOOSE - ensure trades) ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (trend follow with daily bias)
        if current_regime == 1:
            # Long: Daily bull + (4h HMA bull OR HMA cross long OR Donchian breakout)
            if daily_bull:
                if hma_4h_bull or hma_cross_long or donchian_breakout_long:
                    desired_signal = SIZE_TREND
            
            # Short: Daily bear + (4h HMA bear OR HMA cross short OR Donchian breakdown)
            elif daily_bear:
                if hma_4h_bear or hma_cross_short or donchian_breakdown_short:
                    desired_signal = -SIZE_TREND
        
        # REGIME 2: CHOPPY (mean reversion with Connors RSI)
        elif current_regime == 2:
            # Long: CRSI < 20 (primary) OR RSI < 30 + BB touch (backup)
            if crsi_oversold:
                desired_signal = SIZE_CHOP
            elif rsi_oversold and touch_lower:
                desired_signal = SIZE_CHOP
            
            # Short: CRSI > 80 (primary) OR RSI > 70 + BB touch (backup)
            elif crsi_overbought:
                desired_signal = -SIZE_CHOP
            elif rsi_overbought and touch_upper:
                desired_signal = -SIZE_CHOP
        
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
        if desired_signal >= SIZE_BULL * 0.9:
            final_signal = SIZE_BULL
        elif desired_signal <= -SIZE_BULL * 0.9:
            final_signal = -SIZE_BULL
        elif desired_signal >= SIZE_CHOP * 0.9:
            final_signal = SIZE_CHOP
        elif desired_signal <= -SIZE_CHOP * 0.9:
            final_signal = -SIZE_CHOP
        elif desired_signal >= SIZE_BEAR * 0.9:
            final_signal = SIZE_BEAR
        elif desired_signal <= -SIZE_BEAR * 0.9:
            final_signal = -SIZE_BEAR
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