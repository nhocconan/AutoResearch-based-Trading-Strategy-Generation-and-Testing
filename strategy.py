#!/usr/bin/env python3
"""
Experiment #011: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Dual Regime

Hypothesis: 4h timeframe with Connors RSI (CRSI) for mean reversion and Donchian 
breakout for trend following, filtered by Choppiness Index regime, should generate
30-60 trades/year with positive Sharpe across all symbols.

Key insight from failures: 
- Simple RSI failed (too many false signals in trends)
- CRSI combines 3 components for better mean reversion timing
- Choppiness Index properly switches between mean revert vs trend follow
- 1w HMA provides macro bias to avoid counter-trend trades

CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long when CRSI < 15 (oversold) + price > SMA200 + weekly bullish
- Short when CRSI > 85 (overbought) + price < SMA200 + weekly bearish

Regime Switch:
- CHOP > 55 = ranging → use CRSI mean reversion
- CHOP < 45 = trending → use Donchian breakout + HMA trend

Position size: 0.30 (discrete)
Stoploss: 2.5*ATR trailing
Target: 30-60 trades/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_donchian_regime_1d1w_v2"
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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentile rank of price change over rank_period
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of streaks
    # Streak = consecutive up (+1) or down (-1) days
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if delta[i-1] > 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if delta[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank of price change
    price_change = close_s.diff()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = price_change.iloc[i-rank_period:i]
        current = price_change.iloc[i]
        rank = np.sum(window < current)
        percent_rank[i] = rank / rank_period * 100.0
    
    # Combine components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
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
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
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
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (highest + lowest) / 2.0
    return highest, lowest, mid

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for intermediate trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_high, donch_low, donch_mid = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    # Also calculate HMA on 4h for trend
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
    
    for i in range(250, n):  # Need 200 for SMA + 50 for HTF alignment
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donch_high[i]) or np.isnan(sma_200[i]) or np.isnan(hma_4h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H TREND ===
        price_above_hma_4h = close[i] > hma_4h[i]
        price_below_hma_4h = close[i] < hma_4h[i]
        
        # === SMA200 LONG-TERM FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range-bound market
        is_trending = chop_value < 45.0  # Trending market
        
        # === CONNORS RSI EXTREMES (Mean Reversion) ===
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        
        # === DONCHIAN BREAKOUT (Trend Following) ===
        donchian_breakout_long = close[i] > donch_high[i-1]  # Break above previous high
        donchian_breakout_short = close[i] < donch_low[i-1]  # Break below previous low
        
        # === VOLATILITY FILTER ===
        atr_7 = calculate_atr(high, low, close, period=7)
        vol_elevated = atr_7[i] > atr_14[i] * 1.15  # Recent vol spike
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with CRSI ---
        if is_ranging:
            # Long: CRSI oversold + price above SMA200 + weekly/4h not strongly bearish
            if crsi_oversold:
                if price_above_sma200 and (price_above_hma_1w or price_above_hma_4h):
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + price below SMA200 + weekly/4h not strongly bullish
            elif crsi_overbought:
                if price_below_sma200 and (price_below_hma_1w or price_below_hma_4h):
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Donchian Breakout ---
        elif is_trending:
            # Long: Donchian breakout + all trend filters bullish
            if donchian_breakout_long:
                if price_above_hma_1w and price_above_hma_1d and price_above_hma_4h:
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakout + all trend filters bearish
            elif donchian_breakout_short:
                if price_below_hma_1w and price_below_hma_1d and price_below_hma_4h:
                    new_signal = -POSITION_SIZE
        
        # --- NEUTRAL REGIME (45-55 CHOP): Use 4h HMA crossover ---
        else:
            # Long: Price crosses above HMA4h + weekly confirms
            if close[i] > hma_4h[i] and close[i-1] <= hma_4h[i-1]:
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE
            
            # Short: Price crosses below HMA4h + weekly confirms
            elif close[i] < hma_4h[i] and close[i-1] >= hma_4h[i-1]:
                if price_below_hma_1w:
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
        
        # === EXIT ON REGIME/TREND CHANGE ===
        # Exit long if weekly trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_1w and price_below_hma_4h:
                new_signal = 0.0
        
        # Exit short if weekly trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_1w and price_above_hma_4h:
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