#!/usr/bin/env python3
"""
Experiment #448: 4h Regime-Adaptive Strategy with Weekly HMA Bias + Choppiness Filter
Hypothesis: Market regime (trending vs ranging) determines which strategy works.
Choppiness Index (CHOP) detects regime: CHOP<38.2=trending, CHOP>61.8=ranging.
In trending regime: follow weekly HMA bias with 4h pullback entries (RSI 35-45 long, 55-65 short).
In ranging regime: mean revert on RSI extremes (RSI<25 long, RSI>75 short).
Weekly HMA provides strong HTF trend bias. Simpler entry logic = fewer false signals.
3*ATR stoploss for 4h timeframe avoids premature exits. Discrete sizing (0.0, ±0.25, ±0.35).
Timeframe: 4h (REQUIRED), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_chop_weekly_hma_rsi_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market.
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i - period:i + 1])
        lowest = np.min(low[i - period:i + 1])
        
        if highest > lowest:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr = max(high[j] - low[j], 
                         abs(high[j] - close[j - 1]), 
                         abs(low[j] - close[j - 1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        
        upper_band = hl2 + multiplier * atr[i]
        lower_band = hl2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            trend[i] = 1
        else:
            if trend[i - 1] == 1:
                if close[i] < supertrend[i - 1]:
                    supertrend[i] = upper_band
                    trend[i] = -1
                else:
                    supertrend[i] = max(upper_band, supertrend[i - 1])
                    trend[i] = 1
            else:
                if close[i] > supertrend[i - 1]:
                    supertrend[i] = lower_band
                    trend[i] = 1
                else:
                    supertrend[i] = min(lower_band, supertrend[i - 1])
                    trend[i] = -1
    
    return supertrend, trend

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_EXIT = 0.0
    
    # Track position state
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(st_trend[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Regime detection
        trending_regime = chop[i] < 45.0  # Slightly relaxed from 38.2
        ranging_regime = chop[i] > 55.0   # Slightly relaxed from 61.8
        
        # Supertrend direction
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 50
        rsi_pullback_short = rsi[i] > 50 and rsi[i] < 65
        
        new_signal = 0.0
        
        # === TRENDING REGIME: Follow weekly bias with pullback entries ===
        if trending_regime:
            # Long: Weekly bullish + Supertrend bullish + RSI pullback
            if weekly_bullish and st_bullish and rsi_pullback_long:
                new_signal = SIZE_ENTRY
            # Short: Weekly bearish + Supertrend bearish + RSI pullback
            elif weekly_bearish and st_bearish and rsi_pullback_short:
                new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME: Mean revert on RSI extremes ===
        elif ranging_regime:
            # Long: RSI oversold (fade the bottom)
            if rsi_oversold:
                new_signal = SIZE_ENTRY
            # Short: RSI overbought (fade the top)
            elif rsi_overbought:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME: Follow Supertrend only ===
        else:
            if st_bullish and rsi[i] > 45:
                new_signal = SIZE_ENTRY
            elif st_bearish and rsi[i] < 55:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Trailing stoploss at 3*ATR for 4h
            current_stop = entry_price + 3.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = SIZE_EXIT
            else:
                # Update trailing stop (only move up for longs)
                new_trailing = close[i] - 3.0 * atr[i]
                if new_trailing > stoploss_price:
                    stoploss_price = new_trailing
        
        elif position_side < 0 and entry_price > 0:
            # Trailing stoploss at 3*ATR for 4h
            if close[i] > stoploss_price:
                new_signal = SIZE_EXIT
            else:
                # Update trailing stop (only move down for shorts)
                new_trailing = close[i] + 3.0 * atr[i]
                if new_trailing < stoploss_price:
                    stoploss_price = new_trailing
        
        # === UPDATE POSITION STATE ===
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stoploss_price = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stoploss_price = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            stoploss_price = 0.0
        
        signals[i] = new_signal
    
    return signals