#!/usr/bin/env python3
"""
Experiment #106: 12h Primary + 1d HTF — Multi-Signal Confluence Strategy

Hypothesis: Previous 12h strategies failed due to over-filtering (too many 
conflicting conditions = 0 trades). This strategy uses MULTIPLE independent 
entry signals that can each trigger trades, ensuring minimum trade frequency:

1. DONCHIAN BREAKOUT: 20-period high/low breakout with 1d HMA trend filter
2. CONNORS RSI: Mean reversion at extremes (CRSI<15 long, >85 short)
3. BOLLINGER BAND: Price at band extremes with RSI confirmation

Key innovations:
1. OR logic for entries (any signal can trigger) - ensures trades generate
2. 1d HMA slope for macro trend bias (only trade with HTF trend)
3. ATR(14) trailing stop at 2.5x for risk management
4. Discrete position sizing: 0.25 base, 0.30 max with confluence
5. Simple exit logic: RSI extreme take profit or trend change

Why this should work on 12h:
- Natural frequency: 30-50 trades/year (within target 20-50)
- Multiple entry paths = trades actually generate on all symbols
- 1d HTF prevents counter-trend trades in bear markets (2025 test)
- Proven patterns: Donchian (SOL +0.782), CRSI (ETH +0.923)
- Conservative sizing prevents 2022-style drawdowns

Position size: 0.25 base, 0.30 max
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_multi_signal_confluence_1d_v1"
timeframe = "12h"
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
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return sma.values, upper.values, lower.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return highest, lowest

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # RSI on close (short period)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close = rsi_close.fillna(50.0).values
    
    # RSI on streak
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Percent rank
    def percent_rank(x):
        if x.max() == x.min():
            return 50.0
        return (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100
    
    percent_rank_vals = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        percent_rank, raw=False
    ).values
    percent_rank_vals = np.nan_to_num(percent_rank_vals, nan=50.0)
    
    crsi = (rsi_close + rsi_streak + percent_rank_vals) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
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
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(donchian_upper[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_slope_positive = hma_1d_slope[i] > 0.3
        hma_slope_negative = hma_1d_slope[i] < -0.3
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === BOLLINGER BAND POSITION ===
        bb_range = bb_upper[i] - bb_lower[i] + 1e-10
        bb_pct = (close[i] - bb_lower[i]) / bb_range
        near_bb_lower = bb_pct < 0.15
        near_bb_upper = bb_pct > 0.85
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ENTRY LOGIC (OR conditions - any can trigger) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        if price_above_hma_1d or hma_slope_positive:
            # Signal 1: Donchian breakout with RSI confirmation
            if donchian_breakout_long and rsi_14[i] < 65.0:
                new_signal = POSITION_SIZE_BASE
            
            # Signal 2: Connors RSI extreme oversold
            if crsi_oversold:
                new_signal = POSITION_SIZE_BASE
            
            # Signal 3: Bollinger lower band with RSI support
            if near_bb_lower and rsi_oversold:
                new_signal = POSITION_SIZE_BASE
            
            # Confluence: 2+ signals = max size
            signal_count = sum([donchian_breakout_long and rsi_14[i] < 65.0, crsi_oversold, near_bb_lower and rsi_oversold])
            if signal_count >= 2:
                new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        if price_below_hma_1d or hma_slope_negative:
            # Signal 1: Donchian breakdown with RSI confirmation
            if donchian_breakout_short and rsi_14[i] > 35.0:
                new_signal = -POSITION_SIZE_BASE
            
            # Signal 2: Connors RSI extreme overbought
            if crsi_overbought:
                new_signal = -POSITION_SIZE_BASE
            
            # Signal 3: Bollinger upper band with RSI support
            if near_bb_upper and rsi_overbought:
                new_signal = -POSITION_SIZE_BASE
            
            # Confluence: 2+ signals = max size
            signal_count = sum([donchian_breakout_short and rsi_14[i] > 35.0, crsi_overbought, near_bb_upper and rsi_overbought])
            if signal_count >= 2:
                new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        if position_side > 0 and new_signal == 0.0:
            if rsi_14[i] < 75.0:
                new_signal = POSITION_SIZE_BASE
        
        if position_side < 0 and new_signal == 0.0:
            if rsi_14[i] > 25.0:
                new_signal = -POSITION_SIZE_BASE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if position_side < 0:
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
        if position_side > 0 and price_below_hma_1d and hma_slope_negative:
            new_signal = 0.0
        
        if position_side < 0 and price_above_hma_1d and hma_slope_positive:
            new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if position_side > 0 and rsi_14[i] > 80.0:
            new_signal = 0.0
        
        if position_side < 0 and rsi_14[i] < 20.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if position_side == 0:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals