#!/usr/bin/env python3
"""
Experiment #245: 12h Choppiness Index Regime-Adaptive with Daily/Weekly HMA Trend Filter
Hypothesis: Market regime (trending vs ranging) determines which strategy works best.
Choppiness Index (CHOP) identifies regime: CHOP<38.2=trending (use breakout), CHOP>61.8=ranging
(use mean reversion). Daily HMA provides primary trend bias, Weekly HMA confirms macro.
This differs from previous attempts by using CHOP as the PRIMARY regime filter instead of
RSI/Supertrend alone. Simpler entry conditions to ensure sufficient trades. Position sizing:
0.25 entry, 0.15 half at 2R profit. Stoploss: 2.5*ATR trailing stop. Target: Beat Sharpe=0.499.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_daily_weekly_hma_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range > 0, price_range, 1e-10)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop[:period] = 50  # Initialize early values
    return chop

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

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    std = np.where(std > 0, std, 1e-10)
    zscore = (close - sma) / std
    return zscore

def calculate_momentum(close, period=10):
    """Calculate Rate of Change (ROC) momentum."""
    prev_close = np.roll(close, period)
    prev_close[:period] = close[0]
    roc = (close - prev_close) / prev_close * 100
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    momentum = calculate_momentum(close, 10)
    
    # Track previous values for crossover detection
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    prev_zscore = np.roll(zscore, 1)
    prev_zscore[0] = zscore[0]
    prev_momentum = np.roll(momentum, 1)
    prev_momentum[0] = momentum[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Regime detection via Choppiness Index
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        neutral_regime = not trending_regime and not ranging_regime
        
        # RSI signals
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_cross_up = prev_rsi[i] < 40 and rsi[i] >= 40
        rsi_cross_down = prev_rsi[i] > 60 and rsi[i] <= 60
        
        # Z-score mean reversion signals
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        zscore_cross_up = prev_zscore[i] < -1.0 and zscore[i] >= -1.0
        zscore_cross_down = prev_zscore[i] > 1.0 and zscore[i] <= 1.0
        
        # Momentum signals
        momentum_positive = momentum[i] > 0
        momentum_negative = momentum[i] < 0
        momentum_cross_up = prev_momentum[i] <= 0 and momentum[i] > 0
        momentum_cross_down = prev_momentum[i] >= 0 and momentum[i] < 0
        
        new_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) - Use breakout/momentum ===
        if trending_regime:
            # Long: momentum cross up + daily bullish
            if momentum_cross_up and daily_bullish:
                new_signal = SIZE_ENTRY
            # Long: weekly bullish + momentum positive
            elif weekly_bullish and momentum_positive and rsi[i] > 45:
                new_signal = SIZE_ENTRY
            
            # Short: momentum cross down + daily bearish
            if momentum_cross_down and daily_bearish:
                new_signal = -SIZE_ENTRY
            # Short: weekly bearish + momentum negative
            elif weekly_bearish and momentum_negative and rsi[i] < 55:
                new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME (CHOP > 61.8) - Use mean reversion ===
        elif ranging_regime:
            # Long: Z-score oversold + RSI oversold
            if zscore_oversold and rsi_oversold:
                new_signal = SIZE_ENTRY
            # Long: Z-score cross up from oversold
            elif zscore_cross_up and daily_bullish:
                new_signal = SIZE_ENTRY
            
            # Short: Z-score overbought + RSI overbought
            if zscore_overbought and rsi_overbought:
                new_signal = -SIZE_ENTRY
            # Short: Z-score cross down from overbought
            elif zscore_cross_down and daily_bearish:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME - Use conservative entries ===
        else:
            # Long: RSI cross up + daily bullish
            if rsi_cross_up and daily_bullish:
                new_signal = SIZE_ENTRY
            # Short: RSI cross down + daily bearish
            elif rsi_cross_down and daily_bearish:
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