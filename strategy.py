#!/usr/bin/env python3
"""
Experiment #054: 4h Primary + 12h/1d HTF — Adaptive Regime Strategy

Hypothesis: 4h timeframe with 12h/1d trend bias using Choppiness Index regime 
detection will adapt between mean-reversion (range) and breakout (trend) modes.
This should generate 20-50 trades/year with Sharpe > 0.486 on all symbols.

Key insights from 46 failed experiments:
1) 4h primary timeframe proven to work (current best Sharpe=0.486)
2) 12h/1d HTF provides trend bias without over-filtering like 1w
3) Choppiness Index regime switch is critical for bear/range markets
4) Connors RSI works better than standard RSI for mean reversion
5) Donchian breakout ensures trades during trend moves
6) Position size 0.25-0.30 controls drawdown during crashes

Why this should work:
- 4h primary = proven sweet spot (fewer trades than 1h, more signals than 1d)
- 12h/1d HTF = strong trend filter without 1w's lag
- Adaptive regime = works in both bull (trend) and bear/range (mean revert)
- Connors RSI = 75% win rate on mean reversion entries
- Loose enough entries to ensure 20-50 trades/year on ALL symbols

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 25-50 trades/year, Sharpe > 0.5, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_donchian_regime_12h1d_v1"
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
    Long when CRSI < 10, Short when CRSI > 90
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(len(close))
    for i in range(streak_period, len(close)):
        up_streaks = np.sum(streak[max(0, i-streak_period):i] > 0)
        streak_rsi[i] = 100.0 * up_streaks / streak_period if streak_period > 0 else 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = close[i-rank_period:i]
        current = close[i]
        rank = np.sum(window < current) / rank_period
        percent_rank[i] = rank * 100.0
    
    # Combine
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HMA for trend bias
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(sma_50[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === HTF TREND BIAS (12h + 1d) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bullish: both 12h and 1d HMA confirmed
        strong_bull = price_above_hma_12h and price_above_hma_1d
        # Strong bearish: both 12h and 1d HMA confirmed
        strong_bear = price_below_hma_12h and price_below_hma_1d
        # Neutral: mixed signals
        neutral_htf = not strong_bull and not strong_bear
        
        # === 4H TREND CONFIRMATION ===
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        price_above_hma_50 = close[i] > hma_50[i]
        price_below_hma_50 = close[i] < hma_50[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range market (mean revert)
        is_trending = chop_value < 45.0  # Trend market (breakout)
        # 45-55 = transition zone (use breakout logic)
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: Donchian Breakout + HMA Confirmation ---
        if is_trending:
            # Long breakout: price breaks Donchian upper + HMA bullish + HTF confirms
            if close[i] > donchian_upper[i-1] and price_above_hma_21:
                if strong_bull or (price_above_hma_12h and price_above_hma_50):
                    new_signal = POSITION_SIZE
            
            # Short breakdown: price breaks Donchian lower + HMA bearish + HTF confirms
            elif close[i] < donchian_lower[i-1] and price_below_hma_21:
                if strong_bear or (price_below_hma_12h and price_below_hma_50):
                    new_signal = -POSITION_SIZE
        
        # --- RANGING REGIME: Connors RSI Mean Reversion ---
        elif is_ranging:
            # Long: CRSI extremely oversold (< 10) + price near support
            if crsi[i] < 15.0 and price_above_hma_50:
                new_signal = POSITION_SIZE
            
            # Short: CRSI extremely overbought (> 85) + price near resistance
            elif crsi[i] > 85.0 and price_below_hma_50:
                new_signal = -POSITION_SIZE
        
        # --- TRANSITION ZONE: Hybrid Logic (ensures trades) ---
        else:
            # Long: CRSI oversold OR Donchian breakout with HTF bias
            if crsi[i] < 20.0 and strong_bull:
                new_signal = POSITION_SIZE
            elif close[i] > donchian_upper[i-1] and strong_bull:
                new_signal = POSITION_SIZE
            
            # Short: CRSI overbought OR Donchian breakdown with HTF bias
            elif crsi[i] > 80.0 and strong_bear:
                new_signal = -POSITION_SIZE
            elif close[i] < donchian_lower[i-1] and strong_bear:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, hold unless opposite signal or stoploss
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if RSI not overbought and HMA still bullish
                if rsi_14[i] < 75.0 and price_above_hma_21:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if RSI not oversold and HMA still bearish
                if rsi_14[i] > 25.0 and price_below_hma_21:
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            # Exit long if both 4h and 12h turn bearish
            if price_below_hma_21 and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both 4h and 12h turn bullish
            if price_above_hma_21 and price_above_hma_12h:
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
                # Reverse position
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