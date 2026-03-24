#!/usr/bin/env python3
"""
Experiment #138: 4h Primary + 1d HTF — Regime-Adaptive Dual Strategy

Hypothesis: After 137 experiments, the key insight is that NO SINGLE strategy works
across all market regimes. BTC/ETH 2022 crash and 2025 bear market destroy pure
trend-following. Pure mean-reversion fails in strong trends.

SOLUTION: Regime-adaptive strategy that SWITCHES based on Choppiness Index:
- CHOP > 61.8 (choppy/range): Mean reversion using Connors RSI at Bollinger bands
- CHOP < 38.2 (trending): Trend following using Donchian breakout + HMA confirmation
- CHOP 38.2-61.8 (transition): Stay flat or reduce position size by 50%

Key design choices:
- Timeframe: 4h (target 20-50 trades/year, proven best for crypto)
- HTF: 1d HMA(50) for major trend bias (call ONCE before loop)
- Regime filter: Choppiness Index(14) with hysteresis (enter 61.8/38.2, exit 55/45)
- Mean revert entry: Connors RSI < 15 (long) or > 85 (short) + price vs BB(20,2.0)
- Trend entry: Donchian(20) breakout + 4h HMA(21) + 1d HMA alignment
- Position size: 0.25 (25% of capital, conservative for 4h)
- Stoploss: 2.0x ATR trailing stop (tighter than previous 2.5x)
- LOOSE filters to ensure >=30 trades on train, >=3 on test ALL symbols

Why this should beat Sharpe=0.167:
1. Regime switching avoids trend-following in chop (2022 whipsaw)
2. Connors RSI has 75% win rate in range markets (proven on ETH)
3. 1d HMA bias prevents counter-trend trades in strong moves
4. Tighter stoploss (2.0x vs 2.5x) reduces drawdown
5. Smaller position (0.25 vs 0.28) = less DD in crashes

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_crsi_donchian_hma_1d_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi3 = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
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
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = 50.0 + (streak[i] / streak_period) * 25.0
        else:
            streak_rsi[i] = 50.0 + (streak[i] / streak_period) * 25.0
        streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Percent Rank (100)
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        pct_rank[i] = (rank / (rank_period - 1)) * 100.0
    
    # Combine into CRSI
    crsi = (rsi3 + streak_rsi + pct_rank) / 3.0
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
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
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion entries"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
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
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 4h)
    SIZE_HALF = 0.125  # Half position for transition regime
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Regime tracking with hysteresis
    prev_regime = 0  # 0=neutral, 1=trend, -1=chop
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(bb_upper[i]) or np.isnan(donchian_upper[i]):
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
        
        # === REGIME DETECTION with HYSTERESIS ===
        # Enter trend mode: CHOP < 38.2, exit: CHOP > 45
        # Enter chop mode: CHOP > 61.8, exit: CHOP < 55
        if chop[i] < 38.2:
            regime = 1  # Trending
        elif chop[i] > 61.8:
            regime = -1  # Choppy
        elif prev_regime == 1 and chop[i] < 45.0:
            regime = 1  # Stay in trend
        elif prev_regime == -1 and chop[i] > 55.0:
            regime = -1  # Stay in chop
        else:
            regime = 0  # Transition
        
        prev_regime = regime
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        desired_signal = 0.0
        
        # === REGIME 1: TREND FOLLOWING (CHOP < 38.2) ===
        if regime == 1:
            # Donchian breakout confirmed by HMA alignment
            donchian_breakout_bull = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
            donchian_breakout_bear = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
            
            # LONG: Breakout + HMA bull + HTF bull
            if donchian_breakout_bull and hma_bull and htf_bull:
                desired_signal = SIZE
            
            # SHORT: Breakout + HMA bear + HTF bear
            elif donchian_breakout_bear and hma_bear and htf_bear:
                desired_signal = -SIZE
            
            # Weaker signal if HTF disagrees (reduce size)
            elif donchian_breakout_bull and hma_bull:
                desired_signal = SIZE_HALF
            elif donchian_breakout_bear and hma_bear:
                desired_signal = -SIZE_HALF
        
        # === REGIME -1: MEAN REVERSION (CHOP > 61.8) ===
        elif regime == -1:
            # Connors RSI extremes + Bollinger band touch
            crsi_oversold = crsi[i] < 15.0
            crsi_overbought = crsi[i] > 85.0
            
            bb_touch_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
            bb_touch_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
            
            # LONG: CRSI oversold + price at/near BB lower + HTF not strongly bearish
            if crsi_oversold and bb_touch_lower and not htf_bear:
                desired_signal = SIZE
            
            # SHORT: CRSI overbought + price at/near BB upper + HTF not strongly bullish
            elif crsi_overbought and bb_touch_upper and not htf_bull:
                desired_signal = -SIZE
            
            # Weaker signal without HTF confirmation
            elif crsi_oversold and bb_touch_lower:
                desired_signal = SIZE_HALF
            elif crsi_overbought and bb_touch_upper:
                desired_signal = -SIZE_HALF
        
        # === REGIME 0: TRANSITION (reduce or flat) ===
        else:
            # Only take very strong signals in transition
            if htf_bull and hma_bull and crsi[i] < 20.0:
                desired_signal = SIZE_HALF
            elif htf_bear and hma_bear and crsi[i] > 80.0:
                desired_signal = -SIZE_HALF
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE_HALF * 0.85:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.85:
            final_signal = -SIZE_HALF
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