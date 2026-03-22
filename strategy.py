#!/usr/bin/env python3
"""
Experiment #604: 4h Primary + 12h HTF — KAMA Adaptive Trend + Choppiness Regime + RSI Timing

Hypothesis: Building on #594 success (KAMA+ADX+CHOP on 4h with 12h HTF, Sharpe=0.465), this 
strategy refines the entry timing with RSI pullback logic and improves regime detection.
Key insight from failures: symmetric long/short rules underperform because crypto markets
are asymmetric (sharp drops, slow rallies). This version uses:

1. KAMA (Kaufman Adaptive MA) - adapts to volatility, less whipsaw than EMA/HMA
2. 12h KAMA for primary trend direction (HTF filter)
3. Choppiness Index for regime: trend-follow when CHOP<45, mean-revert when CHOP>55
4. RSI for entry timing: pullback entries in trend regime, extreme reversals in chop
5. 3*ATR trailing stoploss for risk management

Why this might beat Sharpe=0.520:
- KAMA adapts to market noise better than static MAs (proven in #594)
- 12h HTF trend filter reduces false signals in counter-trend moves
- Regime-switching logic matches market conditions (trend vs range)
- RSI pullback entries (40-55 for longs, 45-60 for shorts) avoid chasing
- Conservative position size (0.30) controls drawdown through 2022 crash

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 20-50 trades/year on 4h (per Rule 10)
Stoploss: 3*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_rsi_12h_v2"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    ER = 1: pure trend (fast EMA), ER = 0: pure noise (slow EMA)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Price change over ER period
    price_change = np.abs(close_s.diff(er_period).values)
    
    # Sum of absolute price changes (volatility)
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    # Efficiency Ratio (ER)
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Fill initial values
    kama[:er_period] = np.nan
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    # Clip to valid range
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h KAMA for trend direction
    kama_12h = calculate_kama(df_12h['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(kama_12h_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS (KAMA slope over 5 bars) ===
        kama_slope_bull = kama_12h_aligned[i] > kama_12h_aligned[i-5] if i >= 5 else False
        kama_slope_bear = kama_12h_aligned[i] < kama_12h_aligned[i-5] if i >= 5 else False
        
        # Price relative to 12h KAMA
        price_above_kama_12h = close[i] > kama_12h_aligned[i]
        price_below_kama_12h = close[i] < kama_12h_aligned[i]
        
        # === 4H KAMA SLOPE (3 bars) ===
        kama_4h_slope_bull = kama_4h[i] > kama_4h[i-3] if i >= 3 else False
        kama_4h_slope_bear = kama_4h[i] < kama_4h[i-3] if i >= 3 else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 45.0  # Trending
        is_chop_regime = chop_14[i] > 55.0   # Choppy/Range
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TREND REGIME: Follow 12h trend with pullback entries ---
        if is_trend_regime:
            # Long: 12h bull slope + 4h bull + price above 12h KAMA + RSI pullback
            if kama_slope_bull and kama_4h_slope_bull and price_above_kama_12h:
                if 40.0 <= rsi_14[i] <= 55.0:
                    new_signal = POSITION_SIZE
            
            # Short: 12h bear slope + 4h bear + price below 12h KAMA + RSI bounce
            elif kama_slope_bear and kama_4h_slope_bear and price_below_kama_12h:
                if 45.0 <= rsi_14[i] <= 60.0:
                    new_signal = -POSITION_SIZE
        
        # --- CHOP REGIME: Mean reversion at RSI extremes ---
        elif is_chop_regime:
            # Long: RSI < 30 (oversold) + price below 12h KAMA (counter-trend)
            if rsi_14[i] < 30.0 and price_below_kama_12h:
                new_signal = POSITION_SIZE
            
            # Short: RSI > 70 (overbought) + price above 12h KAMA (counter-trend)
            elif rsi_14[i] > 70.0 and price_above_kama_12h:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position and no new signal, maintain current position
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        # Exit long if 12h trend flips bear + price below KAMA
        if in_position and position_side > 0:
            if kama_slope_bear and price_below_kama_12h and rsi_14[i] > 50.0:
                new_signal = 0.0
        
        # Exit short if 12h trend flips bull + price above KAMA
        if in_position and position_side < 0:
            if kama_slope_bull and price_above_kama_12h and rsi_14[i] < 50.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals