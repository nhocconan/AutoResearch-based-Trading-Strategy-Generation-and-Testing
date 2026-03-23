#!/usr/bin/env python3
"""
Experiment #649: 4h Primary + 1d HTF — Dual Regime (Choppiness + CRSI/HMA)

Hypothesis: Building on #641 (Sharpe=0.277) which showed CRSI+Choppiness works on 4h,
but current best is 1d timeframe (Sharpe=0.520). This strategy adapts the winning
1d formula to 4h by using 1d Choppiness for regime detection and 1d HMA for trend,
while 4h CRSI provides entry timing precision.

Key insights from failures:
- #644 (4h HMA+RSI+ADX) got Sharpe=-1.995: ADX filter too restrictive
- #639 (4h KAMA+ADX) got Sharpe=-0.112: Too many whipsaws in 2022
- #641 (4h CRSI+Chop) got Sharpe=0.277: POSITIVE! This combination works
- 1d strategies work best because fewer whipsaws

Why this might beat Sharpe=0.520:
- 1d Choppiness regime filter (proven on #641 and current best)
- 1d HMA slope for major trend direction (slower = fewer false signals)
- 4h CRSI for entry timing (faster than 1d, catches pullbacks)
- Dual regime: trend-follow when CHOP<45, mean-revert when CHOP>55
- Simple logic = more trades (avoid 0-trade failure like #638, #645, #648)
- Conservative sizing (0.30) + ATR stop controls drawdown

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 4h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_crsi_chop_1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_high_low = highest_high - lowest_low
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero
    range_high_low = np.where(range_high_low < 1e-10, 1e-10, range_high_low)
    
    chop = 100.0 * np.log10(sum_atr / range_high_low) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Fast RSI on price
    RSI(streak, 2): RSI on up/down streak length
    PercentRank(100): Percentile rank of today's return over last 100 days
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(close, 3)
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Calculate streak
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI on streak (use absolute values for RSI calculation)
    streak_pos = np.maximum(streak, 0)
    streak_neg = np.maximum(-streak, 0)
    
    # Simple RSI on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100): percentile of today's return vs last 100
    returns = close_s.pct_change()
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    
    for i in range(rank_period, n):
        window = returns.iloc[max(0, i-rank_period):i]
        if len(window) > 0:
            current_return = returns.iloc[i]
            if not np.isnan(current_return):
                percent_rank.iloc[i] = (window < current_return).sum() / len(window) * 100.0
    
    percent_rank = percent_rank.fillna(50.0).values
    
    # CRSI = average of three components
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators for regime and trend
    chop_1d = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=50)
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(chop_1d_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(crsi_4h[i]) or np.isnan(atr_14[i]) or np.isnan(hma_4h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D REGIME DETECTION (Choppiness Index) ===
        # CHOP < 45 = trending market (follow trend)
        # CHOP > 55 = ranging market (mean revert)
        # 45-55 = neutral (stay flat or hold existing)
        regime_trending = chop_1d_aligned[i] < 45.0
        regime_ranging = chop_1d_aligned[i] > 55.0
        
        # === 1D TREND DIRECTION (HMA slope over 3 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H CRSI EXTREMES (Connors RSI) ===
        # CRSI < 15 = oversold (long opportunity in uptrend or range)
        # CRSI > 85 = overbought (short opportunity in downtrend or range)
        crsi_oversold = crsi_4h[i] < 15.0
        crsi_overbought = crsi_4h[i] > 85.0
        
        # Moderate CRSI for range entries
        crsi_mild_oversold = crsi_4h[i] < 30.0
        crsi_mild_overbought = crsi_4h[i] > 70.0
        
        # === 4H HMA CONFIRMATION ===
        price_above_hma_4h = close[i] > hma_4h[i]
        price_below_hma_4h = close[i] < hma_4h[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: Follow 1d trend on 4h CRSI pullback ---
        # Long: 1d trending up + price above 1d HMA + 4h CRSI oversold
        if regime_trending and hma_1d_slope_bull and price_above_hma_1d:
            if crsi_oversold or (crsi_mild_oversold and price_above_hma_4h):
                new_signal = POSITION_SIZE
        
        # Short: 1d trending down + price below 1d HMA + 4h CRSI overbought
        elif regime_trending and hma_1d_slope_bear and price_below_hma_1d:
            if crsi_overbought or (crsi_mild_overbought and price_below_hma_4h):
                new_signal = -POSITION_SIZE
        
        # --- RANGING REGIME: Mean revert on CRSI extremes ---
        # Long: 1d ranging + 4h CRSI very oversold
        elif regime_ranging:
            if crsi_oversold:
                new_signal = POSITION_SIZE
            # Short: 1d ranging + 4h CRSI very overbought
            elif crsi_overbought:
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
        
        # === EXIT ON REGIME/TREND FLIP ===
        # Exit long if trend flips bearish in trending regime
        if in_position and position_side > 0:
            if regime_trending and hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        # Exit short if trend flips bullish in trending regime
        if in_position and position_side < 0:
            if regime_trending and hma_1d_slope_bull and price_above_hma_1d:
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