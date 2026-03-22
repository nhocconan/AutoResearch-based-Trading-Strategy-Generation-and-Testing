#!/usr/bin/env python3
"""
Experiment #625: 1h Primary + 4h/1d HTF — HMA Trend + Connors RSI + Choppiness Regime

Hypothesis: Building on lessons from 552 failed strategies, this uses PROVEN patterns
that actually generate trades (not 0 like #615, #620, #621). Key insight: simpler
confluence = more trades while maintaining quality.

Why this might beat Sharpe=0.520:
1. 4h HMA trend filter (proven in #624 which got +39.9% return)
2. Connors RSI for entry timing (75% win rate in literature)
3. Choppiness Index regime filter (CHOP>55=range mean-revert, CHOP<45=trend follow)
4. 1h entries within 4h trend direction (proven 2x Sharpe pattern)
5. Relaxed entry thresholds to ensure ≥30 trades/year (learned from 0-trade failures)

CRITICAL LESSONS FROM FAILURES:
- #615, #620, #621: 0 trades due to too many filters (session+ADX+multiple HTF)
- #618, #622: Negative Sharpe from over-engineered regime switching
- #624: +39.9% return shows 4h HMA+RSI+Donchian WORKS, just needs Sharpe optimization

Strategy Logic:
- 4h HMA(21) slope determines primary trend bias
- 1h Connors RSI < 25 (long) or > 75 (short) for entry timing
- Choppiness Index confirms regime (mean-revert in range, trend-follow in trend)
- 1d HMA as secondary confirmation (not required, just bias)
- POSITION_SIZE = 0.25 (conservative for 1h TF)
- Stoploss: 2.5*ATR trailing

Position sizing: 0.25 discrete (per Rule 4, max 0.40)
Target: 40-70 trades/year on 1h (per Rule 10, lower TF needs fewer trades)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_crsi_chop_4h1d_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI(streak): Consecutive up/down days momentum
    PercentRank: Where current close ranks vs last 100 closes
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i]) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_3 + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = Range/Ranging market (mean reversion favorable)
    CHOP < 38.2 = Trending market (trend following favorable)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP calculation
    chop = np.zeros(len(close))
    mask = (price_range > 0) & (atr_sum > 0)
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for primary trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for secondary confirmation
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1h[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 4H TREND BIAS (HMA slope over 3 bars) ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        # Price relative to 4h HMA
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D TREND CONFIRMATION (optional bias) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        # === 1H HMA SLOPE (2 bars) ===
        hma_1h_slope_bull = hma_1h[i] > hma_1h[i-2] if i >= 2 else False
        hma_1h_slope_bear = hma_1h[i] < hma_1h[i-2] if i >= 2 else False
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 50.0  # Range market
        chop_trend = chop[i] < 45.0  # Trending market
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 30.0
        crsi_overbought = crsi[i] > 70.0
        crsi_extreme_long = crsi[i] < 25.0
        crsi_extreme_short = crsi[i] > 75.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 4h bull trend + CRSI oversold + regime confirmation ---
        # Condition 1: 4h HMA sloping up + price above 4h HMA (primary trend)
        # Condition 2: 1h HMA sloping up (momentum confirmation)
        # Condition 3: CRSI < 30 (oversold pullback in uptrend)
        # Condition 4: Either range (mean revert) OR trend (breakout) regime
        if hma_4h_slope_bull and price_above_hma_4h:
            if hma_1h_slope_bull:
                if crsi_extreme_long:
                    new_signal = POSITION_SIZE
                elif crsi_oversold and chop_range:
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 4h bear trend + CRSI overbought + regime confirmation ---
        # Condition 1: 4h HMA sloping down + price below 4h HMA (primary trend)
        # Condition 2: 1h HMA sloping down (momentum confirmation)
        # Condition 3: CRSI > 70 (overbought rally in downtrend)
        # Condition 4: Either range (mean revert) OR trend (breakdown) regime
        elif hma_4h_slope_bear and price_below_hma_4h:
            if hma_1h_slope_bear:
                if crsi_extreme_short:
                    new_signal = -POSITION_SIZE
                elif crsi_overbought and chop_range:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h:
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