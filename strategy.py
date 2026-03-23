#!/usr/bin/env python3
"""
Experiment #094: 4h Primary + 12h/1d HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: Previous 4h strategies failed due to overly strict entry conditions or wrong regime logic.
This version uses Connors RSI (proven 75% win rate) with Choppiness regime switching.

Key innovations:
1) Connors RSI (CRSI) for entry timing - combines RSI(3) + Streak RSI + PercentRank
2) Choppiness Index determines regime: CHOP>55 = mean revert, CHOP<45 = trend follow
3) 12h HMA for intermediate trend bias (not too strict)
4) 1d HMA slope for macro filter (only blocks extreme counter-trend)
5) Loose CRSI thresholds (10-25 for long, 75-90 for short) to ensure trades
6) ATR(14) trailing stoploss at 2.5x

Why this should work:
- CRSI proven in bear markets (2022 crash, 2025 test)
- Regime switching allows both mean reversion AND trend following
- 4h TF naturally limits to 20-50 trades/year
- MTF structure from proven strategies (#079, current best)
- Loose thresholds ensure ≥10 trades per symbol

Position size: 0.25 base, 0.35 max with confluence
Stoploss: 2.5*ATR trailing
Target: 25-45 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_12h1d_v1"
timeframe = "4h"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down day streaks
    PercentRank: Where current close ranks in last N bars (0-100)
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak_abs[i] == 0:
            streak_rsi[i] = 50.0
        else:
            # Map streak length to 0-100 scale
            # Longer streak = more extreme
            if streak[i] > 0:
                streak_rsi[i] = min(100, 50 + streak_abs[i] * 25)
            else:
                streak_rsi[i] = max(0, 50 - streak_abs[i] * 25)
    
    # PercentRank - where does current close rank in last N bars?
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = (rank / rank_period) * 100.0
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    crsi[:max(rsi_period, streak_period, rank_period)] = 50.0  # warmup
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_sma(close, period=200):
    """Calculate SMA."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HMA for intermediate trend
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d HMA slope (trend strength)
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    ema_21 = calculate_ema(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MED = 0.30
    POSITION_SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(ema_21[i]):
            continue
        
        # === HTF TREND BIAS (12h HMA + 1d HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        hma_1d_slope_positive = hma_1d_slope[i] > 0.05
        hma_1d_slope_negative = hma_1d_slope[i] < -0.05
        
        # === CHOPPINESS REGIME ===
        chop_ranging = chop_14[i] > 55.0  # mean reversion regime
        chop_trending = chop_14[i] < 45.0  # trend following regime
        chop_neutral = not chop_ranging and not chop_trending
        
        # === CRSI EXTREMES (LOOSE thresholds for trade generation) ===
        crsi_oversold = crsi[i] < 25.0  # long signal (looser than 10)
        crsi_overbought = crsi[i] > 75.0  # short signal (looser than 90)
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === EMA/SMA CONFIRMATION ===
        ema_bullish = close[i] > ema_21[i]
        ema_bearish = close[i] < ema_21[i]
        price_above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        price_below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Mean Reversion (Ranging Regime) ---
        # CRSI oversold + price near/above key levels
        if chop_ranging and crsi_oversold:
            # Must have some trend support (not fighting 1d trend too hard)
            if price_above_hma_12h or (price_below_hma_12h and not hma_1d_slope_negative):
                new_signal = POSITION_SIZE_BASE
                # Boost if extreme CRSI
                if crsi_extreme_oversold:
                    new_signal = POSITION_SIZE_MED
        
        # --- LONG ENTRY: Trend Following (Trending Regime) ---
        # Pullback in uptrend + CRSI not too overbought
        if chop_trending and price_above_hma_12h and price_above_hma_1d:
            if crsi[i] < 50.0 and ema_bullish:  # pullback in uptrend
                new_signal = POSITION_SIZE_BASE
                if crsi_oversold:
                    new_signal = POSITION_SIZE_MED
        
        # --- SHORT ENTRY: Mean Reversion (Ranging Regime) ---
        # CRSI overbought + price near/below key levels
        if chop_ranging and crsi_overbought:
            # Must have some trend support (not fighting 1d trend too hard)
            if price_below_hma_12h or (price_above_hma_12h and not hma_1d_slope_positive):
                new_signal = -POSITION_SIZE_BASE
                # Boost if extreme CRSI
                if crsi_extreme_overbought:
                    new_signal = -POSITION_SIZE_MED
        
        # --- SHORT ENTRY: Trend Following (Trending Regime) ---
        # Rally in downtrend + CRSI not too oversold
        if chop_trending and price_below_hma_12h and price_below_hma_1d:
            if crsi[i] > 50.0 and ema_bearish:  # rally in downtrend
                new_signal = -POSITION_SIZE_BASE
                if crsi_overbought:
                    new_signal = -POSITION_SIZE_MED
        
        # --- NEUTRAL REGIME: Use SMA200 filter ---
        if chop_neutral:
            if crsi_oversold and price_above_sma200:
                new_signal = POSITION_SIZE_BASE
            elif crsi_overbought and price_below_sma200:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Keep position if CRSI hasn't reached extreme exit zone
        if in_position and new_signal == 0.0:
            if position_side > 0 and crsi[i] < 80.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi[i] > 20.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if 12h HMA turns bearish strongly
        if in_position and position_side > 0:
            if price_below_hma_12h and hma_1d_slope_negative:
                new_signal = 0.0
        
        # Exit short if 12h HMA turns bullish strongly
        if in_position and position_side < 0:
            if price_above_hma_12h and hma_1d_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 85.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 15.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals