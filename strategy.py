#!/usr/bin/env python3
"""
Experiment #384: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime

Hypothesis: Previous strategies failed due to overly complex regime detection
(ADX + Choppiness together rarely triggered). This version uses SIMPLER regime
detection (Choppiness only) + Connors RSI for entries (proven 75% win rate).

Key changes from #352:
1. Remove ADX from regime detection (too many false negatives)
2. Use Connors RSI instead of regular RSI (better mean reversion signal)
3. Loosen entry thresholds significantly to ensure trades generate
4. Add 1w HTF for ultra-long-term bias (reduces whipsaw in bear markets)
5. Simpler position tracking with ATR-based stoploss

Regime Detection (SIMPLIFIED):
- CHOP > 55 = choppy/range → mean reversion entries (CRSI extremes)
- CHOP < 45 = trending → breakout entries (HMA alignment)
- Between = use previous regime (hysteresis)

Entry Logic:
- Choppy Long: CRSI < 15 + price > HMA(50) + 1d HMA bull
- Choppy Short: CRSI > 85 + price < HMA(50) + 1d HMA bear
- Trending Long: HMA(21) bull + 1d bull + 1w bull + price > Donchian(20) high
- Trending Short: HMA(21) bear + 1d bear + 1w bear + price < Donchian(20) low

Position sizing: 0.25 base, 0.30 when all HTF aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_hma_1d1w_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - combines RSI, streak RSI, and percentile rank"""
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
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
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        up_streaks = np.sum(streak_vals > 0)
        total_streaks = np.sum(np.abs(streak_vals))
        if total_streaks > 0:
            streak_rsi[i] = 100.0 * up_streaks / total_streaks
        else:
            streak_rsi[i] = 50.0
    
    # Percentile Rank(100) - where current price ranks in last 100 bars
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        pct_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # CRSI = average of three components
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
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
        
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(crsi[i]):
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
        
        # === REGIME DETECTION (Choppiness only - simpler) ===
        # CHOP > 55 = choppy/range
        # CHOP < 45 = trending
        # Between = use previous regime (hysteresis)
        
        if not np.isnan(chop[i]):
            if chop[i] > 55.0:
                current_regime = 2  # choppy
            elif chop[i] < 45.0:
                current_regime = 1  # trending
            else:
                current_regime = prev_regime
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === HTF BIAS (1d and 1w) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_21[i]
        hma_bear = close[i] < hma_21[i]
        
        above_hma50 = close[i] > hma_50[i]
        below_hma50 = close[i] < hma_50[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = False
        breakout_short = False
        if not np.isnan(donchian_upper[i-1]):
            breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            breakout_short = close[i] < donchian_lower[i-1]
        
        # === CRSI EXTREMES (LOOSENED for more trades) ===
        crsi_oversold = crsi[i] < 20.0  # Very oversold
        crsi_overbought = crsi[i] > 80.0  # Very overbought
        
        # === ENTRY LOGIC (SIMPLIFIED - ensure trades generate) ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (breakout + trend alignment)
        if current_regime == 1:
            # Long: HMA bull + 1d bull + breakout (3 conditions max)
            if hma_bull and htf_1d_bull and breakout_long:
                # Add 1w for stronger signal
                if htf_1w_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Short: HMA bear + 1d bear + breakdown (3 conditions max)
            elif hma_bear and htf_1d_bear and breakout_short:
                # Add 1w for stronger signal
                if htf_1w_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # REGIME 2: CHOPPY (CRSI mean reversion - VERY SIMPLE)
        elif current_regime == 2:
            # Long: CRSI oversold + above HMA50 (just 2 conditions!)
            if crsi_oversold and above_hma50:
                desired_signal = SIZE_BASE
            
            # Short: CRSI overbought + below HMA50 (just 2 conditions!)
            elif crsi_overbought and below_hma50:
                desired_signal = -SIZE_BASE
        
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