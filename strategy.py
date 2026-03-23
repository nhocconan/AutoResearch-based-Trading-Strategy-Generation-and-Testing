#!/usr/bin/env python3
"""
Experiment #340: 1h Primary + 4h/12h HTF — Regime-Adaptive Mean Reversion with Trend Bias

Hypothesis: Previous 1h/4h strategies failed because:
1. Entry conditions too strict → 0 trades (see #328, #330)
2. Symmetric long/short logic fails in bear markets (BTC -25% in 2025)
3. Too many regime filters killing all signals

This strategy uses:
1. 12h HMA(21) as MACRO BIAS (hard filter: prefer longs if bullish, shorts if bearish)
2. 4h HMA(16/48) for trend direction (proven in best strategy mtf_hma_rsi_zscore_v1)
3. 1h Connors RSI for entry timing (CRSI < 25 long, > 75 short) — LOOSENED for trade count
4. 1h Choppiness Index regime (CHOP > 55 = mean revert, CHOP < 45 = trend follow)
5. Session filter: Only 8-20 UTC (high volume hours) — reduces false signals
6. Volume confirmation: volume > 0.8x 20-bar avg (not 1.5x which is too strict)
7. ATR trailing stop (2.5x ATR) for risk management

KEY INSIGHT: Use HTF (4h/12h) for DIRECTION, 1h only for ENTRY TIMING.
This gives HTF trade frequency (30-60/year) with 1h execution precision.
LOOSENED CRSI thresholds (< 25 / > 75 instead of < 10 / > 90) to ensure trades.

TARGET: 30-60 trades/year on 1h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
POSITION SIZE: 0.25 (smaller for 1h to minimize fee drag from more frequent signals)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_regime_4h12h_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness.
    """
    close_s = pd.Series(close)
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_n)
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    Values > 61.8 = choppy/range, < 38.2 = trending
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

def calculate_rsi(close, period):
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: % of past 100 days with lower returns
    
    Entry: CRSI < 10 (extreme oversold), CRSI > 90 (extreme overbought)
    We use < 25 / > 75 for more trades.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(close, 3)
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (period 2)
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        count_lower = (window < returns.iloc[i]).sum()
        percent_rank[i] = 100.0 * count_lower / rank_period
    
    # CRSI
    with np.errstate(invalid='ignore'):
        crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def get_hour_from_open_time(open_time_arr):
    """Extract hour (0-23) from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time_arr // (1000 * 3600)) % 24
    return hours

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
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume moving average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for trend bias
    # 4h HMA(16/48) for intermediate trend
    hma_4h_fast_raw = calculate_hma(df_4h['close'].values, 16)
    hma_4h_slow_raw = calculate_hma(df_4h['close'].values, 48)
    
    hma_4h_fast = align_htf_to_ltf(prices, df_4h, hma_4h_fast_raw)
    hma_4h_slow = align_htf_to_ltf(prices, df_4h, hma_4h_slow_raw)
    
    # 12h HMA(21) for macro bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Extract hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h (smaller to reduce fee drag)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_fast[i]) or np.isnan(hma_4h_slow[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_ma_20[i]
        
        # === MACRO BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h[i]
        price_below_hma_12h = close[i] < hma_12h[i]
        
        # === INTERMEDIATE TREND (4h HMA crossover) ===
        hma_4h_bullish = hma_4h_fast[i] > hma_4h_slow[i]
        hma_4h_bearish = hma_4h_fast[i] < hma_4h_slow[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === CONNORS RSI SIGNALS (LOOSENED for trade count) ===
        crsi_oversold = crsi[i] < 25.0  # Long entry
        crsi_overbought = crsi[i] > 75.0  # Short entry
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Only trade during session hours with volume confirmation
        if in_session and volume_confirmed:
            if is_choppy:
                # CHOPPY REGIME: Mean reversion with HTF bias
                # Long: CRSI oversold + price above 12h HMA (bullish macro)
                if crsi_oversold and price_above_hma_12h:
                    desired_signal = BASE_SIZE
                
                # Short: CRSI overbought + price below 12h HMA (bearish macro)
                elif crsi_overbought and price_below_hma_12h:
                    desired_signal = -BASE_SIZE
            
            elif is_trending:
                # TREND REGIME: Follow 4h trend on pullbacks
                # Long: 4h bullish + CRSI pullback (not extreme, 30-50)
                if hma_4h_bullish and 30 <= crsi[i] <= 50:
                    desired_signal = BASE_SIZE
                
                # Short: 4h bearish + CRSI pullback (50-70)
                elif hma_4h_bearish and 50 <= crsi[i] <= 70:
                    desired_signal = -BASE_SIZE
            
            else:
                # NEUTRAL REGIME: Use 4h trend direction only
                if hma_4h_bullish and crsi_oversold:
                    desired_signal = BASE_SIZE
                elif hma_4h_bearish and crsi_overbought:
                    desired_signal = -BASE_SIZE
        
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
        
        # === CRSI EXTREME EXIT (reversal signal) ===
        if in_position and position_side > 0 and crsi[i] > 80:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if trend still valid
            if position_side > 0 and hma_4h_bullish:
                desired_signal = BASE_SIZE
            elif position_side < 0 and hma_4h_bearish:
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