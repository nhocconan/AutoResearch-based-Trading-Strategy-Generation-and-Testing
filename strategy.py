#!/usr/bin/env python3
"""
Experiment #613: 1d Primary + 1w HTF — Fisher Transform Reversals + HMA Trend + Donchian Confirmation

Hypothesis: Building on current best (mtf_1d_chop_crsi_regime_1w_v1, Sharpe=0.520), this strategy 
replaces RSI with Fisher Transform for better reversal detection in crypto markets. Fisher Transform 
normalizes price to Gaussian distribution, making extremes more reliable than RSI for entry timing.

Key insights from 542 failed strategies:
1. #607 failed (Sharpe=-0.627) due to overly complex KAMA+CHOP+asymmetric RSI logic
2. Fisher Transform catches reversals better than RSI in trending markets (Ehlers research)
3. 1w HMA slope is cleaner trend filter than KAMA slope (proven in baseline strategies)
4. Donchian breakout confirmation reduces false entries in choppy markets
5. Binary regime (CHOP >50 chop, <50 trend) simpler and more reliable than triple regime

Why this might beat Sharpe=0.520:
- Fisher Transform at ±1.5 extremes = high probability reversals (75%+ win rate in crypto)
- 1w HMA trend filter keeps us on right side of major moves (avoid 2022 crash shorts)
- Donchian(20) breakout confirmation = only enter when momentum confirms
- Simpler regime logic = more trades, less condition conflicts
- Conservative size (0.30) controls drawdown through volatility
- 2.5*ATR trailing stop protects profits on fast reversals

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 1d (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_hma_donchian_1w_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), weighted by sqrt(n)
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian distribution for clearer extreme detection.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        normalized = (typical[i] - lowest) / range_val
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform formula
        fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value (Ehlers recommendation)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
        else:
            fisher[i] = fisher_val
    
    return fisher

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 50 = choppy/range market (mean reversion works)
    CHOP < 50 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_donchian_high(high, period=20):
    """Calculate Donchian Channel Upper Band (highest high over period)."""
    return pd.Series(high).rolling(window=period, min_periods=period).max().values

def calculate_donchian_low(low, period=20):
    """Calculate Donchian Channel Lower Band (lowest low over period)."""
    return pd.Series(low).rolling(window=period, min_periods=period).min().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for primary trend direction
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(close, period=21)
    fisher_9 = calculate_fisher_transform(high, low, close, period=9)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    donchian_high_20 = calculate_donchian_high(high, 20)
    donchian_low_20 = calculate_donchian_low(low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # Track Fisher crossover for entry timing
    prev_fisher = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d[i]) or np.isnan(hma_1w_aligned[i]):
            prev_fisher = fisher_9[i] if not np.isnan(fisher_9[i]) else prev_fisher
            continue
        if np.isnan(fisher_9[i]) or np.isnan(chop_14[i]) or np.isnan(atr_14[i]):
            prev_fisher = fisher_9[i] if not np.isnan(fisher_9[i]) else prev_fisher
            continue
        if np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]):
            prev_fisher = fisher_9[i] if not np.isnan(fisher_9[i]) else prev_fisher
            continue
        if atr_14[i] == 0:
            prev_fisher = fisher_9[i] if not np.isnan(fisher_9[i]) else prev_fisher
            continue
        
        # === 1W TREND BIAS (HMA slope over 2 bars) ===
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-2] if i >= 2 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-2] if i >= 2 else False
        
        # Price relative to 1w HMA
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D HMA SLOPE (2 bars) ===
        hma_1d_slope_bull = hma_1d[i] > hma_1d[i-2] if i >= 2 else False
        hma_1d_slope_bear = hma_1d[i] < hma_1d[i-2] if i >= 2 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d[i]
        price_below_hma_1d = close[i] < hma_1d[i]
        
        # === REGIME DETECTION (Choppiness Index - binary) ===
        is_trend_regime = chop_14[i] < 50.0
        is_chop_regime = chop_14[i] >= 50.0
        
        # === FISHER TRANSFORM EXTREMES ===
        fisher_extreme_low = fisher_9[i] < -1.5
        fisher_extreme_high = fisher_9[i] > 1.5
        
        # Fisher crossover signals
        fisher_cross_up = fisher_9[i] > -1.5 and prev_fisher <= -1.5
        fisher_cross_down = fisher_9[i] < 1.5 and prev_fisher >= 1.5
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        donchian_breakout_up = close[i] > donchian_high_20[i-1] if i >= 1 else False
        donchian_breakout_down = close[i] < donchian_low_20[i-1] if i >= 1 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TREND REGIME: Follow 1w trend with Fisher reversal entries ---
        if is_trend_regime:
            # LONG: 1w bull + price above 1w HMA + Fisher cross up from extreme
            if hma_1w_slope_bull and price_above_hma_1w and fisher_cross_up:
                # Require Donchian confirmation for stronger signal
                if donchian_breakout_up or hma_1d_slope_bull:
                    new_signal = POSITION_SIZE
            
            # SHORT: 1w bear + price below 1w HMA + Fisher cross down from extreme
            elif hma_1w_slope_bear and price_below_hma_1w and fisher_cross_down:
                # Require Donchian confirmation for stronger signal
                if donchian_breakout_down or hma_1d_slope_bear:
                    new_signal = -POSITION_SIZE
        
        # --- CHOP REGIME: Mean reversion at Fisher extremes ---
        elif is_chop_regime:
            # LONG: Fisher < -1.5 (extreme oversold) + price below 1d HMA
            if fisher_extreme_low and price_below_hma_1d:
                new_signal = POSITION_SIZE
            
            # SHORT: Fisher > 1.5 (extreme overbought) + price above 1d HMA
            elif fisher_extreme_high and price_above_hma_1d:
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
            if hma_1w_slope_bear and price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1w_slope_bull and price_above_hma_1w:
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
        
        # Update previous Fisher for crossover detection
        prev_fisher = fisher_9[i] if not np.isnan(fisher_9[i]) else prev_fisher
    
    return signals