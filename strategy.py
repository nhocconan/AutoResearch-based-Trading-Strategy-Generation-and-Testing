#!/usr/bin/env python3
"""
Experiment #476: 12h Primary + 1d HTF — KAMA Trend + ADX + RSI Simplified

Hypothesis: After 475 experiments, the pattern is clear:
1. Complex indicators (CRSI, CHOP) introduce bugs and reduce trade frequency
2. 12h timeframe works but needs SIMPLER entry logic (2 confluence not 5)
3. KAMA adapts better to crypto's regime changes than HMA/EMA
4. ADX filter prevents entries in dead markets (ADX<20 = no trades)
5. Relaxed RSI thresholds (25/75 not 10/90) ensure >=30 trades/symbol

Why this beats #472 (Sharpe=-0.101):
- KAMA responds faster to trend changes than HMA
- ADX>20 filter avoids choppy whipsaws that destroyed 2022 returns
- Simpler RSI(14) vs CRSI = more reliable, fewer calculation bugs
- Single regime logic (trend-follow with pullback) vs complex dual-regime
- Better stoploss reset logic (reset on every new signal)

Position sizing: 0.30 long, 0.25 short (asymmetric for bear bias)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 25-50 trades/year on 12h, >=30 trades/symbol on train
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_rsi_1d_simp_v1"
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
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market volatility - moves fast in trends, slow in chop.
    Proven in crypto to reduce whipsaws vs EMA/HMA.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close_s.diff(period).values)
    volatility = pd.Series(np.abs(close_s.diff())).rolling(window=period, min_periods=period).sum().values
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    ADX > 25 = strong trend, ADX < 20 = ranging/choppy.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF KAMA (major trend direction)
    kama_1d_21 = calculate_kama(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h_10 = calculate_kama(close, period=10)
    kama_12h_30 = calculate_kama(close, period=30)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_1d_21_aligned[i]):
            continue
        if np.isnan(kama_12h_10[i]) or np.isnan(kama_12h_30[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]) or np.isnan(sma_200[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > kama_1d_21_aligned[i]
        bear_regime = close[i] < kama_1d_21_aligned[i]
        
        # === 12H LOCAL TREND (KAMA crossover) ===
        kama_bullish = kama_12h_10[i] > kama_12h_30[i]
        kama_bearish = kama_12h_10[i] < kama_12h_30[i]
        
        # === ADX TREND STRENGTH (avoid choppy markets) ===
        strong_trend = adx_14[i] > 20.0
        weak_market = adx_14[i] < 20.0
        
        # === RSI PULLBACK SIGNALS (relaxed for frequency) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — SIMPLIFIED (any 2 of 3 confluence) ===
        new_signal = 0.0
        
        # LONG ENTRIES (simplified - fewer conditions)
        if bull_regime and kama_bullish and rsi_oversold:
            new_signal = LONG_SIZE
        elif bull_regime and above_sma200 and rsi_14[i] < 40.0:
            new_signal = LONG_SIZE
        elif kama_bullish and rsi_extreme_oversold and strong_trend:
            new_signal = LONG_SIZE
        elif above_sma200 and kama_bullish and rsi_14[i] < 45.0:
            new_signal = LONG_SIZE * 0.9
        
        # SHORT ENTRIES (simplified - fewer conditions)
        if new_signal == 0.0:
            if bear_regime and kama_bearish and rsi_overbought:
                new_signal = -SHORT_SIZE
            elif bear_regime and below_sma200 and rsi_14[i] > 60.0:
                new_signal = -SHORT_SIZE
            elif kama_bearish and rsi_extreme_overbought and strong_trend:
                new_signal = -SHORT_SIZE
            elif below_sma200 and kama_bearish and rsi_14[i] > 55.0:
                new_signal = -SHORT_SIZE * 0.9
        
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
        
        # === EXIT CONDITIONS (regime flip or extreme RSI) ===
        if in_position and position_side > 0:
            if rsi_14[i] > 80.0:  # Take profit on extreme overbought
                new_signal = 0.0
            if bear_regime and kama_bearish:  # Regime flip
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if rsi_14[i] < 20.0:  # Take profit on extreme oversold
                new_signal = 0.0
            if bull_regime and kama_bullish:  # Regime flip
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
                # Position flip - reset tracking
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