#!/usr/bin/env python3
"""
Experiment #002: 12h Primary + 1d HTF — KAMA Adaptive Trend + Choppiness Regime + RSI Entries

Hypothesis: After reviewing failed strategies, the pattern shows:
1. CHOP+CRSI combinations fail when overfiltered (too many conditions)
2. 12h timeframe should target 20-50 trades/year with stricter entries than 1h
3. KAMA (Kaufman Adaptive) outperforms EMA/HMA in mixed regimes (2022 crash + 2025 bear)
4. Choppiness Index correctly switches between trend-follow and mean-revert modes
5. 1d HMA provides reliable trend bias without being too slow

This strategy uses:
- KAMA(10) on 12h for adaptive trend (adjusts to volatility automatically)
- Choppiness Index(14) for regime detection (>61.8=range, <38.2=trend)
- 1d HMA(21) for primary trend direction
- RSI(14) for entry timing with relaxed thresholds (30/70 not 20/80)
- ATR(14) trailing stoploss at 2.5x

Why this might work:
- KAMA flattens in chop (reduces whipsaw), trends in volatility (captures moves)
- CHOP regime filter prevents trend-following in ranges (major loss source)
- 1d HMA gives directional bias without overfiltering
- Relaxed RSI thresholds ensure sufficient trade generation (Rule 9)
- Position size 0.30 discrete minimizes fee churn

Position sizing: 0.30 discrete (conservative for 12h TF)
Target: 25-45 trades/year on 12h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_rsi_1d_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    
    KAMA adapts to market volatility:
    - High volatility (strong trend) → KAMA follows price closely
    - Low volatility (chop) → KAMA flattens (reduces whipsaw)
    
    Efficiency Ratio (ER) = |price change| / sum(|price changes|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = prior_KAMA + SC * (price - prior_KAMA)
    """
    close_s = pd.Series(close)
    
    # Price change over period
    price_change = np.abs(close_s - close_s.shift(period))
    
    # Sum of absolute price changes (volatility)
    vol_sum = pd.Series([np.sum(np.abs(close_s.iloc[max(0,i-period):i+1].diff())) 
                         for i in range(len(close_s))])
    
    # Efficiency Ratio (0 to 1)
    er = price_change / (vol_sum + 1e-10)
    er = er.fillna(0).clip(0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[period] = close_s.iloc[period]
    
    for i in range(period + 1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Range-bound market (mean-reversion favored)
    - CHOP < 38.2: Trending market (trend-following favored)
    - 38.2 < CHOP < 61.8: Transition zone
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Sum of ATR over period
    atr_sum = atr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
    chop = chop.clip(0, 100)
    
    return chop.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_10 = calculate_kama(close, period=10)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(kama_10[i]) or np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPININESS REGIME ===
        chop_val = chop_14[i]
        is_range_regime = chop_val > 61.8
        is_trend_regime = chop_val < 38.2
        
        # === KAMA TREND ===
        kama_slope_bull = kama_10[i] > kama_10[i-3] if i >= 3 else False
        kama_slope_bear = kama_10[i] < kama_10[i-3] if i >= 3 else False
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        # === RSI SIGNALS (relaxed thresholds for trade generation) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_neutral = 35.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market - mean reversion at oversold
        if is_range_regime:
            if rsi_oversold and price_above_hma_1d:
                new_signal = POSITION_SIZE
        
        # Regime 2: Trending market - trend pullback entry
        elif is_trend_regime:
            if hma_1d_slope_bull and price_above_hma_1d:
                if (rsi_oversold or (price_below_kama and kama_slope_bull)):
                    new_signal = POSITION_SIZE
        
        # Regime 3: Transition - KAMA crossover with RSI confirmation
        else:
            if kama_slope_bull and price_above_kama and rsi_14[i] > 45:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market - mean reversion at overbought
        if is_range_regime:
            if rsi_overbought and price_below_hma_1d:
                new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market - trend pullback entry
        elif is_trend_regime:
            if hma_1d_slope_bear and price_below_hma_1d:
                if (rsi_overbought or (price_above_kama and kama_slope_bear)):
                    new_signal = -POSITION_SIZE
        
        # Regime 3: Transition - KAMA crossover with RSI confirmation
        else:
            if kama_slope_bear and price_below_kama and rsi_14[i] < 55:
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
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
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