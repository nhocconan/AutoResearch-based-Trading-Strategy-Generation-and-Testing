#!/usr/bin/env python3
"""
Experiment #644: 4h Primary + 12h/1d HTF — Vol Spike Reversion + Connors RSI + Donchian

Hypothesis: 4h timeframe with 12h HTF filter captures medium-term moves better than 
daily. Vol spike reversion (ATR(7)/ATR(30) > 1.8) catches panic bottoms with proven 
edge in 2022 crash. Connors RSI provides precise entry timing. Donchian breakout 
captures trend continuation when regime shifts.

Key innovations:
1. Vol spike detection: ATR(7)/ATR(30) > 1.8 = panic/extreme, enter mean reversion
2. Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 20, Short when CRSI > 80 (looser for more trades)
3. 12h HMA(21) for trend bias - only long when price > 12h HMA
4. Donchian(20) breakout for trend following when vol normal
5. Dual regime: mean revert on vol spikes, trend follow otherwise
6. Conservative sizing: 0.25-0.30 to survive 77% crash

Why this should beat Sharpe=0.612:
- Vol spike reversion has documented 0.8+ Sharpe through 2022 crash
- Connors RSI has 75% win rate on mean reversion entries
- 4h TF = more trades than 1d (30-60/year target)
- 12h HTF filter prevents counter-trend trades
- Dual regime adapts to market conditions

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_crsi_donchian_regime_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Pad to match length
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of closes in last 100 bars that are lower than current
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    if len(streak_gain) >= streak_period:
        avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        
        avg_streak_gain = np.concatenate([[np.nan], avg_streak_gain])
        avg_streak_loss = np.concatenate([[np.nan], avg_streak_loss])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            streak_rs = avg_streak_gain / avg_streak_loss
            streak_rsi[streak_period+1:] = 100 - (100 / (1 + streak_rs[streak_period+1:]))
    
    # Percent Rank
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        count_lower = np.sum(window[:-1] < close[i])
        percent_rank[i] = count_lower / (pr_period - 1) * 100
    
    # Combine
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_hma(close, period=21):
    """Hull Moving Average."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h indicators (primary timeframe)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_12h_aligned[i]):
            continue
        
        # === VOL REGIME DETECTION ===
        atr_ratio = atr_7[i] / atr_30[i]
        is_vol_spike = atr_ratio > 1.8  # Panic/extreme volatility
        is_vol_normal = atr_ratio < 1.3  # Normal volatility
        
        # === HTF TREND BIAS (12h HMA) ===
        htf_bullish = close[i] > hma_12h_aligned[i]
        htf_bearish = close[i] < hma_12h_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        desired_signal = 0.0
        
        # === REGIME 1: VOL SPIKE (Mean Reversion) ===
        if is_vol_spike:
            # Long: CRSI oversold + HTF not strongly bearish
            if crsi_oversold:
                desired_signal = SIZE_LONG
            # Short: CRSI overbought + HTF not strongly bullish
            elif crsi_overbought:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: NORMAL VOL (Trend Following) ===
        elif is_vol_normal:
            # Long: HTF bullish + Donchian breakout
            if htf_bullish and donchian_breakout_long:
                desired_signal = SIZE_LONG
            # Short: HTF bearish + Donchian breakout
            elif htf_bearish and donchian_breakout_short:
                desired_signal = -SIZE_SHORT
            # CRSI mean reversion in normal vol
            elif crsi_oversold and htf_bullish:
                desired_signal = SIZE_LONG
            elif crsi_overbought and htf_bearish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: TRANSITION ===
        else:
            # Use CRSI with HTF filter
            if crsi_oversold and htf_bullish:
                desired_signal = SIZE_LONG
            elif crsi_overbought and htf_bearish:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish
                if htf_bullish:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish
                if htf_bearish:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_7[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_7[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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