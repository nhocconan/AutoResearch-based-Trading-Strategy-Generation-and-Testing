#!/usr/bin/env python3
"""
Experiment #309: 1h Regime-Adaptive Strategy with Choppiness Index + 4h HMA Bias
Hypothesis: Markets alternate between trending and ranging regimes. By detecting regime
via Choppiness Index (CHOP), we can apply the right strategy: trend-following when CHOP<38.2
(HMA crossover + momentum) and mean-reversion when CHOP>61.8 (RSI extremes + Bollinger).
4h HMA provides macro trend bias to filter counter-trend trades. ATR stops control drawdown.
This adaptive approach should work in both 2021-2024 (bull/bear) and 2025 (bear/range).
Target: Beat Sharpe=0.499 with >=10 trades/symbol, DD > -50%.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_chop_4h_hma_adaptive_atr_v1"
timeframe = "1h"
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
    """Calculate Hull Moving Average for faster trend response."""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range > 0, price_range, 1e-10)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[0:er_period] = np.abs(close[0:er_period] - close[0])
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=er_period, min_periods=er_period).sum().values
    volatility[0] = change[0]
    volatility = np.where(volatility > 0, volatility, 1e-10)
    er = change / volatility
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    kama = calculate_kama(close, 10, 2, 30)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    prev_hma_16 = np.roll(hma_16, 1)
    prev_hma_16[0] = hma_16[0]
    prev_hma_48 = np.roll(hma_48, 1)
    prev_hma_48[0] = hma_48[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(chop[i]) or np.isnan(atr[i]) or np.isnan(bb_upper[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        trending_regime = chop[i] < 45  # Slightly relaxed from 38.2 for more trades
        ranging_regime = chop[i] > 55   # Slightly relaxed from 61.8 for more trades
        neutral_regime = not trending_regime and not ranging_regime
        
        # === MACRO TREND BIAS (4h HMA) ===
        macro_bullish = close[i] > hma_4h_aligned[i]
        macro_bearish = close[i] < hma_4h_aligned[i]
        
        # === TREND FOLLOWING SIGNALS (when trending) ===
        hma_golden_cross = hma_16[i] > hma_48[i] and prev_hma_16[i] <= prev_hma_48[i]
        hma_death_cross = hma_16[i] < hma_48[i] and prev_hma_16[i] >= prev_hma_48[i]
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # === MEAN REVERSION SIGNALS (when ranging) ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_extreme_oversold = rsi[i] < 25
        rsi_extreme_overbought = rsi[i] > 75
        near_bb_lower = close[i] < bb_lower[i] * 1.005
        near_bb_upper = close[i] > bb_upper[i] * 0.995
        rsi_rising = rsi[i] > prev_rsi[i]
        rsi_falling = rsi[i] < prev_rsi[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Trend regime: HMA golden cross + above KAMA + macro bullish
        if trending_regime and hma_golden_cross and above_kama and macro_bullish:
            new_signal = SIZE_ENTRY
        # Trend regime: Above both HMA + macro bullish + RSI momentum
        elif trending_regime and hma_16[i] > hma_48[i] and above_kama and macro_bullish and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Range regime: RSI oversold + near BB lower + macro bullish (counter-trend long)
        elif ranging_regime and rsi_oversold and near_bb_lower and macro_bullish:
            new_signal = SIZE_ENTRY
        # Range regime: RSI extreme oversold + rsi rising (reversal)
        elif ranging_regime and rsi_extreme_oversold and rsi_rising:
            new_signal = SIZE_ENTRY
        # Neutral regime: Simple HMA cross + RSI confirmation
        elif neutral_regime and hma_golden_cross and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Fallback: Macro bullish + above KAMA + RSI > 50 (simple trend)
        elif macro_bullish and above_kama and rsi[i] > 50 and hma_16[i] > hma_48[i]:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Trend regime: HMA death cross + below KAMA + macro bearish
        if trending_regime and hma_death_cross and below_kama and macro_bearish:
            new_signal = -SIZE_ENTRY
        # Trend regime: Below both HMA + macro bearish + RSI momentum
        elif trending_regime and hma_16[i] < hma_48[i] and below_kama and macro_bearish and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Range regime: RSI overbought + near BB upper + macro bearish (counter-trend short)
        elif ranging_regime and rsi_overbought and near_bb_upper and macro_bearish:
            new_signal = -SIZE_ENTRY
        # Range regime: RSI extreme overbought + rsi falling (reversal)
        elif ranging_regime and rsi_extreme_overbought and rsi_falling:
            new_signal = -SIZE_ENTRY
        # Neutral regime: Simple HMA cross + RSI confirmation
        elif neutral_regime and hma_death_cross and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Fallback: Macro bearish + below KAMA + RSI < 50 (simple trend)
        elif macro_bearish and below_kama and rsi[i] < 50 and hma_16[i] < hma_48[i]:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals