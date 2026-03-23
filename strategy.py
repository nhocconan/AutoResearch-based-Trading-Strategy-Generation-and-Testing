#!/usr/bin/env python3
"""
Experiment #359: 4h Primary + 1d HTF — Connors RSI Mean Reversion with Adaptive Regime

Hypothesis: Previous Fisher Transform strategies underperformed because:
1. Fisher extremes (-1.5/+1.5) are too rare on 4h, limiting trade count
2. Binary chop regime (50 threshold) creates whipsaws at boundary
3. 1d HMA as hard filter blocks valid counter-trend mean reversion trades

This strategy uses Connors RSI (CRSI) which is PROVEN for mean reversion:
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long: CRSI < 15 (oversold), Short: CRSI > 85 (overbought)
- More reliable than standard RSI for short-term reversals

KEY IMPROVEMENTS:
1. CRSI instead of Fisher — triggers more frequently with better win rate
2. 3-tier Choppiness: <40 trend, 40-60 transition, >60 range (smoother regime)
3. 1d HMA as BIAS SCALER not hard filter — reduces position size against macro
4. Volume confirmation — only trade when volume > 20-bar average (avoids fakeouts)
5. Simplified exits — CRSI cross through 50 or ATR trail (no complex Fisher exits)
6. RELAXED entry thresholds — CRSI < 20 / > 80 instead of 15/85 for more trades

TARGET: 35-55 trades/year on 4h, Sharpe > 0.6 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_regime_1d_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(streak) measures consecutive up/down days:
    - Streak = number of consecutive days close > prev close (positive) or < (negative)
    - RSI of streak values over 2 periods
    
    PercentRank = percentile of current close within last 100 closes
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI on streak values
    # Streak: count consecutive up/down closes
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-friendly values (absolute with sign preserved for direction)
    streak_for_rsi = np.abs(streak)
    streak_rsi = calculate_rsi(streak_for_rsi, period=streak_period)
    
    # Component 3: Percentile Rank of close over last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # Fill early values
    percent_rank[:rank_period] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_close + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # HMA for trend detection
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    
    # Calculate and align 1d HMA for macro bias (SOFT FILTER - scales position)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 4h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA - SOFT FILTER, scales position) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (3-tier Choppiness) ===
        is_trending = chop[i] < 40.0      # Strong trend
        is_transition = 40.0 <= chop[i] <= 60.0  # Mixed/transition
        is_ranging = chop[i] > 60.0       # Choppy/range
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_avg[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # CRSI extremes for mean reversion entries
        crsi_oversold = crsi[i] < 20.0   # Relaxed from 15 for more trades
        crsi_overbought = crsi[i] > 80.0  # Relaxed from 85 for more trades
        
        # HMA trend confirmation
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === ENTRY LOGIC ===
        if is_ranging or is_transition:
            # RANGE/TRANSITION: Mean reversion with CRSI
            # Long: CRSI < 20 + volume confirmation
            # Short: CRSI > 80 + volume confirmation
            
            if crsi_oversold and volume_confirmed:
                # Scale position by macro bias
                size_mult = 1.0 if price_above_hma_1d else 0.6
                desired_signal = BASE_SIZE * size_mult
            
            elif crsi_overbought and volume_confirmed:
                # Scale position by macro bias
                size_mult = 1.0 if price_below_hma_1d else 0.6
                desired_signal = -BASE_SIZE * size_mult
        
        elif is_trending:
            # TREND: Follow trend but wait for pullback (CRSI extreme)
            # Long: CRSI < 25 (pullback) + HMA bullish + volume
            # Short: CRSI > 75 (rally) + HMA bearish + volume
            
            crsi_pullback_long = crsi[i] < 25.0
            crsi_rally_short = crsi[i] > 75.0
            
            if crsi_pullback_long and hma_bullish and volume_confirmed:
                size_mult = 1.0 if price_above_hma_1d else 0.5
                desired_signal = BASE_SIZE * size_mult
            
            elif crsi_rally_short and hma_bearish and volume_confirmed:
                size_mult = 1.0 if price_below_hma_1d else 0.5
                desired_signal = -BASE_SIZE * size_mult
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === CRSI EXIT (mean reversion complete - cross through 50) ===
        if in_position and position_side > 0 and crsi[i] > 50.0:
            # Long position: exit when CRSI crosses above 50 (mean reached)
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 50.0:
            # Short position: exit when CRSI crosses below 50 (mean reached)
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if regime still valid for position
            if position_side > 0:
                # Long: hold if CRSI still < 50 and not overbought
                if crsi[i] < 55.0:
                    desired_signal = BASE_SIZE * 0.5  # Reduce to half on hold
            elif position_side < 0:
                # Short: hold if CRSI still > 50 and not oversold
                if crsi[i] > 45.0:
                    desired_signal = -BASE_SIZE * 0.5  # Reduce to half on hold
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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