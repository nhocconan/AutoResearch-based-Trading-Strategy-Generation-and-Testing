#!/usr/bin/env python3
"""
Experiment #689: 4h Primary + 1d HTF — KAMA Trend + Donchian Breakout + Regime Filter

Hypothesis: After analyzing failures #684 (-0.292 Sharpe) and #688 (-1.376 Sharpe),
the problem is OVER-FILTERING. CRSI<15 or >85 is TOO RARE. Choppiness regime
switching adds complexity without edge.

This strategy uses SIMPLER, MORE RELIABLE signals:
1. KAMA (Kaufman Adaptive MA) - adapts to volatility, less whipsaw than HMA/EMA
2. Donchian(20) breakout - proven trend entry, generates consistent signals
3. 1d HMA slope - simple trend filter (not complex slope calculations)
4. ATR volatility filter - only trade when vol > median (avoid dead zones)

Key differences from failed #684:
- CRSI threshold relaxed: <25/>75 instead of <15/>85 (MORE TRADES)
- Removed complex regime switching (Chop >55/<45 was too restrictive)
- KAMA instead of HMA (better in choppy markets per literature)
- Donchian breakout for entry timing (proven 60%+ win rate on breakouts)
- Simpler 1d filter: just HMA slope, not price-relative + slope combo

Position sizing: 0.28 discrete (conservative for 4h)
Target: 30-50 trades/year on 4h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_hma_1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    
    KAMA adapts to market volatility:
    - Fast SC (0.6667) when trending (high ER)
    - Slow SC (0.0645) when ranging (low ER)
    
    Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = (ER * (fastSC - slowSC) + slowSC)^2
    KAMA = KAMA[prev] + SC * (Close - KAMA[prev])
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    signal = np.abs(close - np.roll(close, er_period))
    signal[:er_period] = np.nan
    
    noise = np.abs(close - np.roll(close, 1))
    noise_sum = pd.Series(noise).rolling(window=er_period, min_periods=er_period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = signal / (noise_sum + 1e-10)
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

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel.
    Upper = Highest High over period
    Lower = Lowest Low over period
    Middle = (Upper + Lower) / 2
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    
    # ATR median for volatility filter
    atr_median = np.nanmedian(atr_14[100:])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
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
        if np.isnan(kama_4h[i]) or np.isnan(donchian_upper[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0 or atr_median == 0:
            continue
        
        # === 1D TREND BIAS (HMA slope over 3 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        
        # === VOLATILITY FILTER (only trade when vol > 50% of median) ===
        vol_ok = atr_14[i] > 0.5 * atr_median
        
        # === KAMA TREND (4h) ===
        kama_slope_bull = kama_4h[i] > kama_4h[i-2] if i >= 2 else False
        kama_slope_bear = kama_4h[i] < kama_4h[i-2] if i >= 2 else False
        
        price_above_kama = close[i] > kama_4h[i]
        price_below_kama = close[i] < kama_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === RSI FILTER (avoid extreme overbought/oversold on entry) ===
        rsi_not_extreme_long = rsi_14[i] < 75.0  # not too overbought for long
        rsi_not_extreme_short = rsi_14[i] > 25.0  # not too oversold for short
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Condition 1: 1d trend up + 4h KAMA up + Donchian breakout + RSI ok
        if hma_1d_slope_bull and kama_slope_bull and price_above_kama:
            if donchian_breakout_long and rsi_not_extreme_long and vol_ok:
                new_signal = POSITION_SIZE
        
        # Condition 2: 1d trend up + 4h pullback to KAMA + RSI oversold (mean revert in trend)
        elif hma_1d_slope_bull and price_above_kama:
            if rsi_14[i] < 35.0 and vol_ok:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Condition 1: 1d trend down + 4h KAMA down + Donchian breakdown + RSI ok
        if hma_1d_slope_bear and kama_slope_bear and price_below_kama:
            if donchian_breakout_short and rsi_not_extreme_short and vol_ok:
                new_signal = -POSITION_SIZE
        
        # Condition 2: 1d trend down + 4h rally to KAMA + RSI overbought (mean revert in trend)
        elif hma_1d_slope_bear and price_below_kama:
            if rsi_14[i] > 65.0 and vol_ok:
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
            if hma_1d_slope_bear and kama_slope_bear:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and kama_slope_bull:
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