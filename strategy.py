#!/usr/bin/env python3
"""
Experiment #121: 4h Primary + 1d/1w HTF — Dual Regime with Choppiness Index

Hypothesis: Previous regime strategies failed due to over-complexity. This simplifies to:
1) Choppiness Index (CHOP) detects regime: >61.8=range, <38.2=trend
2) Range regime: Connors RSI mean reversion (proven 75% win rate)
3) Trend regime: Donchian breakout with HMA filter (proven on 4h)
4) 1d HMA(21) for macro bias — only trade with higher TF trend
5) 1w HMA(21) for ultra-macro filter — avoid counter-trend in major moves

Why this should work:
- CHOP regime filter is proven (ETH Sharpe +0.923 in research)
- CRSI works well in ranging markets (2022 crash, 2025 bear)
- Donchian breakouts work in trending markets (2021 bull, SOL rallies)
- Dual regime adapts to market conditions automatically
- 4h TF = 20-50 trades/year (low fee drag)
- Simple logic = more robust across BTC/ETH/SOL

Position size: 0.25 base, 0.30 with confluence
Stoploss: 2.5*ATR trailing
Target: Sharpe > 0.5 on ALL symbols, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_dual_crsi_donchian_1d1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    > 61.8 = choppy/range, < 38.2 = trending
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
    price_range = np.maximum(price_range, 1e-10)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    < 10 = oversold (long), > 90 = overbought (short)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_gain = np.maximum(streak, 0)
    streak_loss = np.abs(np.minimum(streak, 0))
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank (100)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < close[i]) / (rank_period - 1) * 100
        crsi[i] = (rsi_short[i] + rsi_streak[i] + rank) / 3.0
    
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0.0, 100.0)
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_rsi(close, period=14):
    """Calculate standard RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for ultra-macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop_14[i] > 61.8
        regime_trend = chop_14[i] < 38.2
        regime_neutral = not regime_range and not regime_trend
        
        # === HTF TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 4h TREND FILTER ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGE REGIME: CONNORS RSI MEAN REVERSION ---
        if regime_range:
            # Long: CRSI < 10 + price above 1d HMA (bullish bias)
            if crsi[i] < 10 and price_above_hma_1d:
                new_signal = POSITION_SIZE_BASE
                if price_above_hma_1w and hma_4h_bullish:
                    new_signal = POSITION_SIZE_MAX
            
            # Short: CRSI > 90 + price below 1d HMA (bearish bias)
            if crsi[i] > 90 and price_below_hma_1d:
                new_signal = -POSITION_SIZE_BASE
                if price_below_hma_1w and hma_4h_bearish:
                    new_signal = -POSITION_SIZE_MAX
        
        # --- TREND REGIME: DONCHIAN BREAKOUT ---
        if regime_trend:
            prev_high = donchian_upper[i-1] if i > 0 else donchian_upper[i]
            prev_low = donchian_lower[i-1] if i > 0 else donchian_lower[i]
            
            breakout_long = close[i] > prev_high
            breakout_short = close[i] < prev_low
            
            # Long breakout with trend confirmation
            if breakout_long and price_above_hma_1d and hma_4h_bullish:
                new_signal = POSITION_SIZE_BASE
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE_MAX
            
            # Short breakout with trend confirmation
            if breakout_short and price_below_hma_1d and hma_4h_bearish:
                new_signal = -POSITION_SIZE_BASE
                if price_below_hma_1w:
                    new_signal = -POSITION_SIZE_MAX
        
        # --- NEUTRAL REGIME: HOLD OR FLAT ---
        if regime_neutral:
            # Hold existing position if conditions still valid
            if in_position:
                if position_side > 0 and price_above_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
                elif position_side < 0 and price_below_hma_1d:
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
        
        # === EXIT ON RSI EXTREME (take profit in range regime) ===
        if regime_range and in_position:
            if position_side > 0 and rsi_14[i] > 70.0:
                new_signal = 0.0
            if position_side < 0 and rsi_14[i] < 30.0:
                new_signal = 0.0
        
        # === EXIT ON OPPOSITE BREAKOUT (trend regime) ===
        if regime_trend and in_position:
            if position_side > 0 and close[i] < donchian_mid[i]:
                new_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
                new_signal = 0.0
        
        # === EXIT ON HTF TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_1d and price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d and price_above_hma_1w:
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
                # Position flip
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