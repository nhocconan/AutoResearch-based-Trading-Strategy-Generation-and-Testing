#!/usr/bin/env python3
"""
Experiment #149: 4h Primary + 1d HTF — Simplified Volatility Breakout

Hypothesis: Previous failures (#141, #139) show complex regime switching = too few trades.
This strategy SIMPLIFIES to proven components with fewer conflicting filters:

1) 1d HMA(21) for macro trend bias — ONLY trade in trend direction
2) ATR Ratio (7/30) for volatility expansion — enter when vol expanding (>1.3)
3) Donchian(20) breakout — classic Turtle Trading entry
4) Fisher Transform(9) for entry timing — catches reversals at breakout
5) ATR(14) trailing stop at 2.5x — protects capital
6) NO complex regime switching — let HTF trend filter handle market conditions

Why this should work:
- Fewer filters = more trades (addressing #141's low trade count)
- ATR ratio captures vol expansion before breakouts (proven in literature)
- Fisher Transform provides precise entry timing at extremes
- 1d HMA bias prevents counter-trend trades (major loss reducer)
- 4h timeframe naturally produces 25-50 trades/year

Position size: 0.25 base, 0.30 with vol confirmation
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_breakout_fisher_1d_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Normalize price to -1 to +1 range
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    price_range = highest - lowest
    price_range = np.maximum(price_range, 1e-10)
    
    normalized = 2.0 * (hl2 - lowest) / price_range - 1.0
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """
    ATR Ratio - measures volatility expansion/contraction.
    Ratio > 1.3 = volatility expanding (good for breakouts)
    Ratio < 0.8 = volatility contracting (avoid trades)
    """
    atr_short = calculate_atr(high, low, close, period=short_period)
    atr_long = calculate_atr(high, low, close, period=long_period)
    
    ratio = atr_short / (atr_long + 1e-10)
    return ratio

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
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(atr_ratio[i]):
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME (ATR Ratio) ===
        vol_expanding = atr_ratio[i] > 1.3
        vol_contracting = atr_ratio[i] < 0.8
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5 and fisher_signal[i] < fisher[i]
        fisher_overbought = fisher[i] > 1.5 and fisher_signal[i] > fisher[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Trend + Breakout + Vol Expansion + Fisher confirmation ---
        if price_above_hma_1d and not price_below_hma_1d:
            # Donchian breakout (price crosses above previous high)
            breakout_long = close[i] > donchian_upper[i-1] and close[i-1] <= donchian_upper[i-1]
            
            if breakout_long and vol_expanding:
                # Fisher confirms (not overbought yet)
                if fisher[i] < 1.5:
                    new_signal = POSITION_SIZE_BASE
                    if atr_ratio[i] > 1.6:
                        new_signal = POSITION_SIZE_MAX
            
            # Pullback entry in uptrend (price near Donchian mid, Fisher oversold)
            elif close[i] > donchian_mid[i] and fisher_oversold:
                new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY: Trend + Breakdown + Vol Expansion + Fisher confirmation ---
        if price_below_hma_1d and not price_above_hma_1d:
            # Donchian breakdown (price crosses below previous low)
            breakdown_short = close[i] < donchian_lower[i-1] and close[i-1] >= donchian_lower[i-1]
            
            if breakdown_short and vol_expanding:
                # Fisher confirms (not oversold yet)
                if fisher[i] > -1.5:
                    new_signal = -POSITION_SIZE_BASE
                    if atr_ratio[i] > 1.6:
                        new_signal = -POSITION_SIZE_MAX
            
            # Pullback entry in downtrend (price near Donchian mid, Fisher overbought)
            elif close[i] < donchian_mid[i] and fisher_overbought:
                new_signal = -POSITION_SIZE_BASE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0 and price_below_hma_1d:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d:
            new_signal = 0.0
        
        # === EXIT ON VOLATILITY CONTRACTION (protect profits) ===
        if in_position and vol_contracting:
            # Reduce position if volatility collapses
            if abs(new_signal) == 0.0:
                new_signal = 0.0
        
        # === EXIT ON FISHER EXTREME (take profit) ===
        if in_position and position_side > 0 and fisher[i] > 2.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -2.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else np.inf
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else np.inf
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = np.inf
        
        signals[i] = new_signal
    
    return signals