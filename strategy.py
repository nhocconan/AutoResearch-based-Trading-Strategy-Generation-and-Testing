#!/usr/bin/env python3
"""
Experiment #530: 30m KAMA Adaptive Crossover with 4h Trend Filter

Hypothesis: After 500+ failed experiments, the key insight is that FIXED-period MAs (EMA/HMA)
fail because they don't adapt to changing volatility. KAMA (Kaufman Adaptive Moving Average)
automatically adjusts smoothing based on market efficiency - smooth in noise, fast in trends.

Why KAMA should work on 30m:
1. 30m has mixed regime (trending + ranging) - KAMA adapts to both
2. 4h KAMA trend filter prevents counter-trend trades (major loss source)
3. Loose RSI filter (30-70) ensures enough trades (>10/symbol/year)
4. 2.0*ATR stoploss appropriate for 30m volatility (tighter than 12h's 2.5*ATR)
5. Position size 0.25 conservative for 30m's higher frequency

Key differences from failed strategies:
- KAMA instead of HMA/EMA (adaptive smoothing)
- Single HTF (4h) not dual (reduces complexity)
- Very loose RSI thresholds (avoid 0-trade problem)
- Simple logic: trend + momentum + not extreme

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_adaptive_4h_trend_rsi_loose_atr_v1"
timeframe = "30m"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    Smooths during noise, responds quickly during trends.
    
    Formula:
    ER = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    close_s = pd.Series(close)
    
    # Change over period
    change = np.abs(close_s - close_s.shift(period))
    
    # Sum of absolute price changes (volatility)
    volatility = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    
    # Efficiency Ratio (0 to 1)
    er = change / volatility.replace(0, np.inf)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    
    # Smoothing Constant (adaptive)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[period-1] = close[period-1]  # Initialize
    
    for i in range(period, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF KAMA for trend direction
    kama_4h = calculate_kama(df_4h['close'].values, period=10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Dual KAMA on 30m (fast/slow for crossover)
    kama_fast = calculate_kama(close, period=6, fast=2, slow=30)
    kama_slow = calculate_kama(close, period=14, fast=2, slow=30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        # === 4h KAMA TREND BIAS ===
        bull_bias = close[i] > kama_4h_aligned[i]
        bear_bias = close[i] < kama_4h_aligned[i]
        
        # === 30m KAMA CROSSOVER ===
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === RSI FILTER (VERY LOOSE to ensure trades) ===
        rsi_ok_long = rsi_14[i] < 70  # Don't buy when extremely overbought
        rsi_ok_short = rsi_14[i] > 30  # Don't sell when extremely oversold
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: 30m KAMA bullish + 4h bullish + RSI not extreme
        if kama_bullish and bull_bias and rsi_ok_long:
            new_signal = SIZE
        
        # Short: 30m KAMA bearish + 4h bearish + RSI not extreme
        elif kama_bearish and bear_bias and rsi_ok_short:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 30m KAMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and kama_bearish:
                new_signal = 0.0
            if position_side < 0 and kama_bullish:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals