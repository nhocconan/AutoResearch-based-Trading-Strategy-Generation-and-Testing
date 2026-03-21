#!/usr/bin/env python3
"""
Experiment #303: 1h Regime-Adaptive Strategy with 4h HMA Trend + Choppiness Index
Hypothesis: Market regime detection (range vs trend) via Choppiness Index allows adaptive logic.
In range regimes (CHOP>61.8): mean reversion with RSI extremes. In trend regimes (CHOP<38.2): trend following with 4h HMA.
This addresses the 2025 bear/range market where pure trend strategies failed.
4h HMA provides macro bias without over-filtering. Simple entry logic ensures >=10 trades.
Position size 0.28 with 2.5*ATR stops controls drawdown while allowing participation.
Target: Beat Sharpe=0.499 from current best while ensuring all symbols have Sharpe>0.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_chop_4h_hma_rsi_adaptive_atr_v1"
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
    CHOP > 61.8 = range/choppy market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # Avoid division by zero
    price_range = np.where(price_range > 0, price_range, 1e-10)
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma, upper, lower

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    std = np.where(std > 0, std, 1e-10)
    zscore = (close - sma) / std
    return zscore

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
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    # Track previous values for crossover detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_hma_21 = np.roll(hma_21, 1)
    prev_hma_21[0] = hma_21[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    prev_chop = np.roll(chop, 1)
    prev_chop[0] = chop[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(atr[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = range/choppy (mean reversion)
        # CHOP < 38.2 = trending (trend following)
        # 38.2 <= CHOP <= 61.8 = transition (use trend bias)
        is_range_regime = chop[i] > 61.8
        is_trend_regime = chop[i] < 38.2
        
        # 4h HMA macro trend bias
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h trend direction
        hma_21_above_50 = hma_21[i] > hma_50[i]
        hma_21_below_50 = hma_21[i] < hma_50[i]
        hma_slope_bullish = hma_21[i] > prev_hma_21[i]
        hma_slope_bearish = hma_21[i] < prev_hma_21[i]
        
        new_signal = 0.0
        
        # === RANGE REGIME: MEAN REVERSION ===
        if is_range_regime:
            # Long: RSI oversold + price near lower BB + zscore <-1.5
            if rsi[i] < 35 and close[i] <= bb_lower[i] and zscore[i] < -1.5:
                new_signal = SIZE_ENTRY
            # Short: RSI overbought + price near upper BB + zscore >1.5
            elif rsi[i] > 65 and close[i] >= bb_upper[i] and zscore[i] > 1.5:
                new_signal = -SIZE_ENTRY
            # Simpler range entries (ensure trades)
            elif rsi[i] < 30 and hma_4h_bullish:
                new_signal = SIZE_ENTRY
            elif rsi[i] > 70 and hma_4h_bearish:
                new_signal = -SIZE_ENTRY
        
        # === TREND REGIME: TREND FOLLOWING ===
        elif is_trend_regime:
            # Long: 4h bullish + 1h bullish + pullback entry
            if hma_4h_bullish and hma_21_above_50 and rsi[i] > 40 and rsi[i] < 60:
                new_signal = SIZE_ENTRY
            # Short: 4h bearish + 1h bearish + pullback entry
            elif hma_4h_bearish and hma_21_below_50 and rsi[i] > 40 and rsi[i] < 60:
                new_signal = -SIZE_ENTRY
            # Breakout entries
            elif hma_4h_bullish and close[i] > hma_21[i] and hma_slope_bullish and rsi[i] > 50:
                new_signal = SIZE_ENTRY
            elif hma_4h_bearish and close[i] < hma_21[i] and hma_slope_bearish and rsi[i] < 50:
                new_signal = -SIZE_ENTRY
        
        # === TRANSITION REGIME: HYBRID ===
        else:
            # Use 4h HMA bias with simpler 1h confirmation
            if hma_4h_bullish and close[i] > hma_21[i] and rsi[i] > 45:
                new_signal = SIZE_ENTRY
            elif hma_4h_bearish and close[i] < hma_21[i] and rsi[i] < 55:
                new_signal = -SIZE_ENTRY
            # Mean reversion in transition
            elif rsi[i] < 28:
                new_signal = SIZE_ENTRY
            elif rsi[i] > 72:
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