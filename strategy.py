#!/usr/bin/env python3
"""
Experiment #009: 1h Dual-Regime Strategy with 4h HMA Trend Filter
Hypothesis: Combine mean-reversion in range markets with trend-following in trending markets.
Uses Choppiness Index (CHOP) to detect regime, then applies appropriate logic:
- CHOP > 55 (range): Mean revert using RSI extremes + BB bands
- CHOP < 45 (trend): Follow 4h HMA direction with pullback entries
4h HMA provides HTF trend bias. Conservative sizing (0.25-0.30) with 2.5*ATR stop.
This addresses failures from pure trend (2022 whipsaw) and pure mean-revert (2025 bear).
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_dual_regime_4h_hma_chop_rsi_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
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
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        tr_range = highest_high - lowest_low
        
        if tr_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / tr_range) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_keltner(high, low, close, atr_period=10, mult=2.0):
    """Calculate Keltner Channel for squeeze detection."""
    atr = calculate_atr(high, low, close, atr_period)
    ema = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    upper = ema + mult * atr
    lower = ema - mult * atr
    return upper, lower, ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_3 = calculate_rsi(close, 3)  # Fast RSI for Connors-style
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    kc_upper, kc_lower, kc_mid = calculate_keltner(high, low, close, 10, 1.5)
    
    # EMA for trend
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Z-score for mean reversion
    rolling_mean = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    rolling_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    zscore = (close - rolling_mean) / (rolling_std + 1e-10)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.28
    SIZE_SHORT = -0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # CHOP > 55 = range/choppy (mean revert)
        # CHOP < 45 = trending (trend follow)
        # 45-55 = neutral (use HTF bias)
        is_range = chop[i] > 55
        is_trend = chop[i] < 45
        
        # 4h HMA trend bias
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # EMA trend
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # Bollinger position
        bb_pct = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # Keltner squeeze (BB inside KC = low vol, breakout imminent)
        bb_inside_kc = (bb_upper[i] < kc_upper[i]) and (bb_lower[i] > kc_lower[i])
        
        new_signal = 0.0
        
        # === RANGE REGIME: MEAN REVERSION ===
        if is_range:
            # Long: RSI oversold + price near BB lower + 4h HMA not strongly bearish
            if rsi[i] < 35 and bb_pct < 0.25 and not hma_4h_bearish:
                new_signal = SIZE_LONG
            # Short: RSI overbought + price near BB upper + 4h HMA not strongly bullish
            elif rsi[i] > 65 and bb_pct > 0.75 and not hma_4h_bullish:
                new_signal = SIZE_SHORT
        
        # === TREND REGIME: TREND FOLLOWING ===
        elif is_trend:
            # Long: 4h HMA bullish + pullback to EMA21 + RSI not overbought
            if hma_4h_bullish and ema_bullish and close[i] < ema_21[i] * 1.01 and rsi[i] < 70:
                new_signal = SIZE_LONG
            # Short: 4h HMA bearish + rally to EMA21 + RSI not oversold
            elif hma_4h_bearish and ema_bearish and close[i] > ema_21[i] * 0.99 and rsi[i] > 30:
                new_signal = SIZE_SHORT
        
        # === NEUTRAL REGIME: HTF BIAS + BREAKOUT ===
        else:
            # Squeeze breakout long
            if bb_inside_kc and close[i] > bb_upper[i] and hma_4h_bullish:
                new_signal = SIZE_LONG
            # Squeeze breakout short
            elif bb_inside_kc and close[i] < bb_lower[i] and hma_4h_bearish:
                new_signal = SIZE_SHORT
            # Z-score extreme mean reversion
            elif zscore[i] < -2.0 and rsi[i] < 40:
                new_signal = SIZE_LONG
            elif zscore[i] > 2.0 and rsi[i] > 60:
                new_signal = SIZE_SHORT
        
        # === STOPLOSS LOGIC (ATR trailing) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals