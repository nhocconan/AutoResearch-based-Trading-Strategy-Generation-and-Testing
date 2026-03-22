#!/usr/bin/env python3
"""
Experiment #020: 30m Ehlers Fisher Transform + 4h HMA Trend + Choppiness Regime
Hypothesis: Fisher Transform catches reversals in bear/range markets where RSI fails.
Research shows Fisher cross above -1.5 (long) and below +1.5 (short) has high win rate.
Combined with 4h HMA for trend bias and Choppiness Index for regime detection.
Choppiness > 61.8 = range (favor mean reversion), < 38.2 = trend (favor breakout).
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Stoploss: 2.5*ATR trailing stop to limit drawdown in volatile crypto markets.
Key innovation: Fisher Transform is superior to RSI for reversal detection in bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_4h_hma_chop_regime_v1"
timeframe = "30m"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    Works exceptionally well in bear/range markets for reversal detection.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    # Calculate median price
    median = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        # Normalize price (avoid division by zero)
        range_val = highest - lowest
        if range_val < 0.0001:
            range_val = 0.0001
        
        # Calculate normalized price
        norm_price = (median[i] - lowest) / range_val
        
        # Apply Fisher Transform formula
        # X = 0.66 * ((price - lowest) / (highest - lowest) - 0.5) + 0.67 * prev_X
        if i > period:
            x = 0.66 * (norm_price - 0.5) + 0.67 * fisher_signal[i - 1]
        else:
            x = 0.66 * (norm_price - 0.5)
        
        # Clamp to avoid extreme values
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher = 0.5 * ln((1 + X) / (1 - X))
        fisher_signal[i] = x
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = ranging market (mean reversion favored)
    CHOP < 38.2 = trending market (trend following favored)
    Values between 38.2-61.8 = transitional
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        price_range = highest - lowest
        if price_range < 0.0001:
            chop[i] = 50.0
            continue
        
        # Sum of ATR over period (approximated as sum of true ranges)
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j - 1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j - 1]) if j > 0 else tr1
            tr_sum += max(tr1, tr2, tr3)
        
        # CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
        if tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Bollinger Bands for additional confirmation
    close_s = pd.Series(close)
    bb_sma = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_sma + 2.0 * bb_std
    bb_lower = bb_sma - 2.0 * bb_std
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    SIZE_EXIT = 0.0
    
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
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Choppiness regime detection
        range_regime = chop[i] > 55.0  # Leaning toward range
        trend_regime = chop[i] < 45.0  # Leaning toward trend
        
        # Fisher Transform signals
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher cross detection (need previous value)
        fisher_cross_up = False
        fisher_cross_down = False
        if i > 0 and not np.isnan(fisher[i - 1]):
            fisher_cross_up = fisher[i - 1] < -1.5 and fisher[i] >= -1.5
            fisher_cross_down = fisher[i - 1] > 1.5 and fisher[i] <= 1.5
        
        # Price position vs Bollinger Bands
        price_near_lower = close[i] < bb_lower[i] * 1.005 if not np.isnan(bb_lower[i]) else False
        price_near_upper = close[i] > bb_upper[i] * 0.995 if not np.isnan(bb_upper[i]) else False
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Fisher cross up from oversold + 4h bull trend
        if fisher_cross_up and bull_trend:
            new_signal = SIZE_MAX
        # Secondary: Fisher oversold + range regime + price above 4h HMA
        elif fisher_oversold and range_regime and bull_trend:
            new_signal = SIZE_BASE
        # Tertiary: Fisher oversold + price near lower BB + EMA bullish
        elif fisher_oversold and price_near_lower and ema_bullish:
            new_signal = SIZE_BASE
        # Quaternary: Fisher cross up + price near lower BB (strong mean reversion)
        elif fisher_cross_up and price_near_lower:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: Fisher cross down from overbought + 4h bear trend
        if fisher_cross_down and bear_trend:
            new_signal = -SIZE_MAX
        # Secondary: Fisher overbought + range regime + price below 4h HMA
        elif fisher_overbought and range_regime and bear_trend:
            new_signal = -SIZE_BASE
        # Tertiary: Fisher overbought + price near upper BB + EMA bearish
        elif fisher_overbought and price_near_upper and ema_bearish:
            new_signal = -SIZE_BASE
        # Quaternary: Fisher cross down + price near upper BB (strong mean reversion)
        elif fisher_cross_down and price_near_upper:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
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
        
        # Short position stoploss
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