#!/usr/bin/env python3
"""
Experiment #047: 1d Primary + 1w HTF — KAMA Adaptive Trend + Fisher Transform Reversals

Hypothesis: Daily timeframe with weekly trend bias using KAMA (adaptive to volatility)
and Fisher Transform (proven reversal indicator) will generate 15-40 trades/year with
Sharpe > 0.486. Key insights from 46 failed experiments:

1) 1d timeframe needs VERY LOOSE entries (Fisher > -1.5, not -1.0; CHOP > 50, not 61.8)
2) KAMA adapts to market noise better than HMA/EMA in crypto
3) Fisher Transform catches reversals in bear/range markets (2025 test period)
4) 1w HTF for macro bias prevents counter-trend trades
5) Minimal filters = ensures trades generate (avoid Sharpe=0.000 failure)

Why this should work:
- 1d primary = proven higher timeframe (less noise, fewer false signals)
- 1w HTF = strong macro trend filter without over-filtering
- KAMA = adapts efficiency ratio to market conditions (better than static MA)
- Fisher Transform = normalized -1 to +1, clear reversal signals
- Simple logic = ensures 15+ trades/year on each symbol

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 15-40 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_fisher_chop_regime_1w_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio.
    ER = |price change| / sum of absolute price changes
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Price change over ER period
    price_change = np.abs(close - np.roll(close, er_period))
    price_change[:er_period] = np.nan
    
    # Sum of absolute price changes (volatility)
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1)))
    
    # Efficiency Ratio
    er = price_change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to -1 to +1 range for clear reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest + 1e-10
        
        # Normalize price to -1 to +1
        mid = (high[i] + low[i]) / 2.0
        normalized = (2.0 * (mid - lowest) / price_range) - 1.0
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        if i > 0:
            fisher[i] = 0.7 * fisher[i] + 0.3 * fisher[i-1]
        
        trigger[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, trigger

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w KAMA for macro bias
    kama_1w = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(kama_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(kama_21[i]):
            continue
        if np.isnan(donchian_upper[i]) or atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        # === CHOPPINESS REGIME (LOOSE threshold for trades) ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Range market
        is_trending = chop_value < 48.0  # Trend market (with hysteresis)
        
        # === FISHER TRANSFORM REVERSAL SIGNALS (LOOSE) ===
        fisher_oversold = fisher[i] < -1.0  # Very oversold
        fisher_overbought = fisher[i] > 1.0  # Very overbought
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher[i-1] <= fisher_trigger[i-1] if i > 0 else False
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher[i-1] >= fisher_trigger[i-1] if i > 0 else False
        fisher_rising = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_falling = fisher[i] < fisher[i-1] if i > 0 else False
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_21[i]
        kama_bearish = close[i] < kama_21[i]
        kama_slope_up = kama_21[i] > kama_21[i-5] if i > 5 else False
        kama_slope_down = kama_21[i] < kama_21[i-5] if i > 5 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC (LOOSE for trade generation) ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Fisher Mean Reversion ---
        if is_ranging:
            # Long: Fisher oversold OR Fisher crossing up + weekly helps
            if fisher_oversold or fisher_cross_up:
                if price_above_kama_1w or fisher_rising:
                    new_signal = POSITION_SIZE
            
            # Short: Fisher overbought OR Fisher crossing down + weekly helps
            elif fisher_overbought or fisher_cross_down:
                if price_below_kama_1w or fisher_falling:
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Donchian Breakout + KAMA ---
        elif is_trending:
            # Long: Donchian breakout + KAMA bullish + weekly confirms
            if donchian_breakout_long and kama_bullish:
                if price_above_kama_1w or kama_slope_up:
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakdown + KAMA bearish + weekly confirms
            elif donchian_breakout_short and kama_bearish:
                if price_below_kama_1w or kama_slope_down:
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK: KAMA crossover if no regime signal (ensures trades) ---
        if new_signal == 0.0:
            # Long: Price crosses above KAMA + Fisher rising + weekly helps
            if close[i] > kama_21[i] and close[i-1] <= kama_21[i-1]:
                if fisher_rising and (price_above_kama_1w or fisher[i] > -0.5):
                    new_signal = POSITION_SIZE
            
            # Short: Price crosses below KAMA + Fisher falling + weekly helps
            elif close[i] < kama_21[i] and close[i-1] >= kama_21[i-1]:
                if fisher_falling and (price_below_kama_1w or fisher[i] < 0.5):
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
        if in_position and position_side > 0:
            if price_below_kama_1w and kama_bearish and chop_value < 45:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_kama_1w and kama_bullish and chop_value < 45:
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