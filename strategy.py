#!/usr/bin/env python3
"""
Experiment #310: 4h Regime-Adaptive Strategy with Daily HMA Bias + Choppiness Filter + KAMA/RSI
Hypothesis: 4h timeframe captures intermediate trends while avoiding 15m/1h noise. 
Use 1d HMA for macro bias, Choppiness Index (CHOP) to detect regime (trending vs ranging).
In trending regime (CHOP<38): use KAMA crossover for entries. In ranging regime (CHOP>62): 
use RSI mean reversion with SMA200 filter. This adaptive approach should work in both 
2021-2024 bull/bear cycles and 2025 bear/range market. Position size 0.30, ATR stop 2.5x.
Target: Beat Sharpe=0.499 with >=10 trades/symbol by having TWO entry modes.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_daily_hma_kama_rsi_atr_v1"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average - adapts to volatility."""
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(period))
    volatility = close_s.diff().abs().rolling(window=period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    return kama

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
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    atr = calculate_atr(high, low, close, period)
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10((highest_high - lowest_low) / (atr * period)) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    kama = calculate_kama(close, 10)
    kama_fast = calculate_kama(close, 5)  # Faster KAMA for crossover
    chop = calculate_choppiness(high, low, close, 14)
    
    # Track previous values for crossover detection
    prev_kama = np.roll(kama, 1)
    prev_kama[0] = kama[0]
    prev_kama_fast = np.roll(kama_fast, 1)
    prev_kama_fast[0] = kama_fast[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
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
    
    for i in range(250, n):  # Need 200 for SMA + 50 buffer
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(sma_200[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Daily macro trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # Regime detection via Choppiness Index
        trending_regime = chop[i] < 45  # Below 45 = trending (use trend strategy)
        ranging_regime = chop[i] > 55   # Above 55 = ranging (use mean reversion)
        
        # Price position relative to SMA200
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # KAMA crossover signals (for trending regime)
        kama_bullish_cross = prev_kama_fast[i] <= prev_kama[i] and kama_fast[i] > kama[i]
        kama_bearish_cross = prev_kama_fast[i] >= prev_kama[i] and kama_fast[i] < kama[i]
        
        # RSI signals (for ranging regime)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_rising = rsi[i] > prev_rsi[i] and rsi[i] < 50
        rsi_falling = rsi[i] < prev_rsi[i] and rsi[i] > 50
        
        # Momentum confirmation
        price_momentum_long = close[i] > prev_close[i] and close[i] > prev_close[max(0, i-5)]
        price_momentum_short = close[i] < prev_close[i] and close[i] < prev_close[max(0, i-5)]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Mode 1: Trending regime + Daily bullish + KAMA bullish cross
        if trending_regime and daily_bullish and kama_bullish_cross:
            new_signal = SIZE_ENTRY
        # Mode 2: Trending regime + Above SMA200 + KAMA fast > KAMA + momentum
        elif trending_regime and above_sma200 and kama_fast[i] > kama[i] and price_momentum_long:
            new_signal = SIZE_ENTRY
        # Mode 3: Ranging regime + Daily bullish + RSI oversold + above SMA200
        elif ranging_regime and daily_bullish and rsi_oversold and above_sma200:
            new_signal = SIZE_ENTRY
        # Mode 4: Ranging regime + RSI rising from oversold + above SMA200
        elif ranging_regime and rsi_rising and above_sma200 and prev_rsi[i] < 40:
            new_signal = SIZE_ENTRY
        # Mode 5: Simple fallback - Daily bullish + Above SMA200 + RSI > 45
        elif daily_bullish and above_sma200 and rsi[i] > 45 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Mode 6: KAMA trend + price above both KAMAs
        elif kama_fast[i] > kama[i] and close[i] > kama_fast[i] and daily_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Mode 1: Trending regime + Daily bearish + KAMA bearish cross
        if trending_regime and daily_bearish and kama_bearish_cross:
            new_signal = -SIZE_ENTRY
        # Mode 2: Trending regime + Below SMA200 + KAMA fast < KAMA + momentum
        elif trending_regime and below_sma200 and kama_fast[i] < kama[i] and price_momentum_short:
            new_signal = -SIZE_ENTRY
        # Mode 3: Ranging regime + Daily bearish + RSI overbought + below SMA200
        elif ranging_regime and daily_bearish and rsi_overbought and below_sma200:
            new_signal = -SIZE_ENTRY
        # Mode 4: Ranging regime + RSI falling from overbought + below SMA200
        elif ranging_regime and rsi_falling and below_sma200 and prev_rsi[i] > 60:
            new_signal = -SIZE_ENTRY
        # Mode 5: Simple fallback - Daily bearish + Below SMA200 + RSI < 55
        elif daily_bearish and below_sma200 and rsi[i] < 55 and rsi[i] > 30:
            new_signal = -SIZE_ENTRY
        # Mode 6: KAMA trend + price below both KAMAs
        elif kama_fast[i] < kama[i] and close[i] < kama_fast[i] and daily_bearish:
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