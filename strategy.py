#!/usr/bin/env python3
"""
Experiment #087: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI + Donchian)

Hypothesis: Single-regime strategies fail because crypto alternates between trending and ranging.
This implements DUAL REGIME logic proven in research:
1) CHOPPINESS INDEX determines regime: CHOP>55 = range (mean revert), CHOP<45 = trend (breakout)
2) RANGE REGIME: Connors RSI <15 long, >85 short (75% win rate in ranges)
3) TREND REGIME: Donchian(20) breakout + 1w HMA trend confirmation
4) 1w HMA provides macro bias (only long if price>1w HMA, only short if below)

Why this should work:
- Dual regime adapts to market conditions (unlike single-regime failures #076, #083, #086)
- Connors RSI proven for mean reversion (ETH Sharpe +0.923 in research)
- Donchian breakout proven for trends (SOL Sharpe +0.782 in research)
- 1d timeframe naturally limits trades to 10-30/year (low fee drag)
- 1w HTF prevents counter-trend trades in bear markets (2025 test)

Position size: 0.25 base, 0.32 max with confluence
Stoploss: 2.5*ATR trailing
Target: 15-35 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_crsi_donchian_1w_v1"
timeframe = "1d"
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
    Proven 75% win rate for mean reversion entries.
    """
    close_s = pd.Series(close)
    
    # RSI(3) - very short term momentum
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (2) - consecutive up/down days
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(len(close))
    for i in range(streak_period, len(close)):
        up_streaks = np.sum(streak[max(0,i-streak_period):i] > 0)
        streak_rsi[i] = 100.0 * up_streaks / streak_period if streak_period > 0 else 50.0
    
    # Percent Rank (100) - where current close ranks in last 100 days
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = close[i-rank_period:i]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine into CRSI
    crsi = (rsi_short.fillna(50.0).values + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (breakout)
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.32
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(sma_200[i]):
            continue
        
        # === HTF MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_ranging = chop_14[i] > 55.0  # range market (mean revert)
        chop_trending = chop_14[i] < 45.0  # trend market (breakout)
        
        # === SMA200 FILTER (long-term bias) ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES (for range regime) ===
        crsi_oversold = crsi[i] < 15.0  # extreme oversold
        crsi_overbought = crsi[i] > 85.0  # extreme overbought
        
        # === DONCHIAN BREAKOUT (for trend regime) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # breakout above previous high
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # breakout below previous low
        
        # === RSI CONFIRMATION ===
        rsi_neutral_long = rsi_14[i] < 60.0
        rsi_neutral_short = rsi_14[i] > 40.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGE REGIME: Mean Reversion with Connors RSI ---
        if chop_ranging:
            # Long: CRSI extreme oversold + price above 1w HMA (macro bullish)
            if crsi_oversold and price_above_hma_1w and rsi_neutral_long:
                new_signal = POSITION_SIZE_BASE
                # Boost if also above SMA200
                if price_above_sma200:
                    new_signal = POSITION_SIZE_MAX
            
            # Short: CRSI extreme overbought + price below 1w HMA (macro bearish)
            if crsi_overbought and price_below_hma_1w and rsi_neutral_short:
                new_signal = -POSITION_SIZE_BASE
                # Boost if also below SMA200
                if price_below_sma200:
                    new_signal = -POSITION_SIZE_MAX
        
        # --- TREND REGIME: Donchian Breakout with HTF Confirmation ---
        if chop_trending:
            # Long: Donchian breakout + price above 1w HMA + above SMA200
            if donchian_breakout_long and price_above_hma_1w and price_above_sma200:
                new_signal = POSITION_SIZE_MAX
            
            # Short: Donchian breakout + price below 1w HMA + below SMA200
            if donchian_breakout_short and price_below_hma_1w and price_below_sma200:
                new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Keep position if no exit signal
        if in_position and new_signal == 0.0:
            if position_side > 0 and rsi_14[i] < 75.0 and crsi[i] < 85.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 25.0 and crsi[i] > 15.0:
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if regime switches from range to trend bearish
        if in_position and position_side > 0:
            if price_below_hma_1w and chop_trending:
                new_signal = 0.0
        
        # Exit short if regime switches from range to trend bullish
        if in_position and position_side < 0:
            if price_above_hma_1w and chop_trending:
                new_signal = 0.0
        
        # === EXIT ON RSI/CRSI EXTREME (take profit) ===
        if in_position and position_side > 0:
            if rsi_14[i] > 75.0 or crsi[i] > 85.0:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if rsi_14[i] < 25.0 or crsi[i] < 15.0:
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