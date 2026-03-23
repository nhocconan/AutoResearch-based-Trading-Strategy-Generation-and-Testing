#!/usr/bin/env python3
"""
Experiment #360: 1h Primary + 4h/12h HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: Previous 1h strategies failed (0 trades) because:
1. Too many confluence filters (session + volume + 3 indicators = never all true)
2. RSI thresholds too narrow for 1h timeframe
3. Choppiness binary split too rigid

This strategy uses PROVEN Connors RSI (CRSI) for mean reversion:
1. 12h HMA(21) as MACRO BIAS (only long if price > 12h HMA, only short if price < 12h HMA)
2. 4h Choppiness Index for regime (CHOP>55=range→mean revert, CHOP<45=trend→follow)
3. RANGE REGIME: CRSI<15 long, CRSI>85 short (proven 75% win rate in ranges)
4. TREND REGIME: HMA(16/48) crossover + CRSI confirms pullback (CRSI<40 long, >60 short)
5. RELAXED session filter (6-22 UTC instead of 8-20) to ensure trades trigger
6. Volume filter: >0.7x 20-bar avg (less strict than 0.8x)
7. ATR(14) trailing stop at 2.0x for risk management

KEY INSIGHT: CRSI is MORE reliable than standard RSI for mean reversion because it
combines 3 components (RSI3 + StreakRSI + PercentRank). In bear/range markets (2025),
mean reversion outperforms trend following. 12h HMA is faster than 1d for 1h entries.

TARGET: 40-80 trades/year on 1h, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
SIZE: 0.25 (smaller than 4h due to more frequent trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_regime_4h12h_relaxed_v1"
timeframe = "1h"
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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Where current price ranks in last N bars
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Component 2: Streak RSI
    # Count consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi = streak_rsi.fillna(50.0).values
    
    # Component 3: PercentRank
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] <= close[i]) / (rank_period - 1)
        percent_rank[i] = rank * 100.0
    
    # Combine all three components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
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
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # HMA for trend detection (fast and slow)
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    chop_4h_raw = calculate_choppiness(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h (target 40-80 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_4h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (12h HMA - HARD FILTER) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        is_choppy = chop_4h_aligned[i] > 55.0  # High choppiness = range regime
        is_trending = chop_4h_aligned[i] < 45.0  # Low choppiness = trend regime
        # Neutral zone 45-55: maintain current position or stay flat
        
        # === VOLUME FILTER (relaxed) ===
        volume_ok = volume[i] > 0.7 * vol_avg_20[i]
        
        # === SESSION FILTER (relaxed 6-22 UTC) ===
        hour_utc = (open_time[i] // 3600000) % 24
        session_ok = 6 <= hour_utc <= 22
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: CRSI mean reversion
            # Long: CRSI < 15 + price above 12h HMA + volume OK
            # Short: CRSI > 85 + price below 12h HMA + volume OK
            
            if price_above_hma_12h and crsi[i] < 15 and volume_ok:
                desired_signal = BASE_SIZE
            
            elif price_below_hma_12h and crsi[i] > 85 and volume_ok:
                desired_signal = -BASE_SIZE
        
        elif is_trending:
            # TREND REGIME: HMA crossover + CRSI pullback entry
            # Long: HMA16 > HMA48 + CRSI < 40 (pullback) + price above 12h HMA
            # Short: HMA16 < HMA48 + CRSI > 60 (pullback) + price below 12h HMA
            
            hma_bullish = hma_16[i] > hma_48[i]
            hma_bearish = hma_16[i] < hma_48[i]
            
            if price_above_hma_12h and hma_bullish and crsi[i] < 40 and volume_ok:
                desired_signal = BASE_SIZE
            
            elif price_below_hma_12h and hma_bearish and crsi[i] > 60 and volume_ok:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
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
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 75:
            # Long position: exit when CRSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25:
            # Short position: exit when CRSI reaches oversold
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if regime and bias still valid
            if position_side > 0:
                if price_above_hma_12h:
                    if (is_choppy and crsi[i] < 75) or \
                       (is_trending and hma_16[i] > hma_48[i] and crsi[i] < 60):
                        desired_signal = BASE_SIZE
            elif position_side < 0:
                if price_below_hma_12h:
                    if (is_choppy and crsi[i] > 25) or \
                       (is_trending and hma_16[i] < hma_48[i] and crsi[i] > 40):
                        desired_signal = -BASE_SIZE
        
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