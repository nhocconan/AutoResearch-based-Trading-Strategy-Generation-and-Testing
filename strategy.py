#!/usr/bin/env python3
"""
Experiment #021: 1h Connors RSI + 4h HMA Trend + Bollinger Regime
Hypothesis: Connors RSI (CRSI) has 75% win rate for mean reversion (research-backed).
Combined with 4h HMA trend filter to avoid counter-trend trades in strong trends.
Bollinger Band width detects regime: narrow = range (favor mean reversion), wide = trend.
ATR trailing stop at 2.0*ATR limits drawdown.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.30) to minimize fee churn.
Key innovation: CRSI combines RSI(3) + StreakRSI(2) + PercentRank(100) for superior entry timing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_bb_regime_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Research shows 75% win rate when CRSI < 10 (long) or > 90 (short).
    Works well in bear/range markets where simple trend strategies fail.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(3)
    rsi3 = np.zeros(n)
    rsi3[:] = np.nan
    for i in range(rsi_period, n):
        gains = np.zeros(i)
        losses = np.zeros(i)
        for j in range(1, i + 1):
            change = close[j] - close[j - 1]
            if change > 0:
                gains[j - 1] = change
            else:
                losses[j - 1] = -change
        
        avg_gain = np.mean(gains[-rsi_period:]) if rsi_period <= i else 0
        avg_loss = np.mean(losses[-rsi_period:]) if rsi_period <= i else 1
        if avg_loss == 0:
            rsi3[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi3[i] = 100 - (100 / (1 + rs))
    
    # Component 2: Streak RSI(2)
    streak = np.zeros(n)
    streak[0] = 0
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1
        else:
            streak[i] = streak[i - 1]
    
    # Convert streak to RSI-like value (absolute streak / period * 100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = np.abs(streak[i - streak_period + 1:i + 1])
        if len(streak_vals) > 0:
            streak_rsi[i] = np.mean(streak_vals) / streak_period * 100
        else:
            streak_rsi[i] = 50.0
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Component 3: Percent Rank(100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = rank / (rank_period - 1) * 100
    
    # Combine components
    valid_mask = (~np.isnan(rsi3)) & (~np.isnan(streak_rsi)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi3[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    return upper, lower, bandwidth, sma

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
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
    
    # Bollinger bandwidth percentile for regime detection
    bb_percentile = pd.Series(bb_bandwidth).rolling(window=100, min_periods=100).apply(
        lambda x: np.sum(x[:-1] < x[-1]) / len(x[:-1]) * 100 if len(x) > 1 else 50
    ).values
    bb_percentile[np.isnan(bb_percentile)] = 50.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Bollinger regime: narrow bandwidth = range (favor mean reversion)
        range_regime = bb_percentile[i] < 40  # Bottom 40% of bandwidth = ranging
        trend_regime = bb_percentile[i] > 60  # Top 40% of bandwidth = trending
        
        # CRSI signals (mean reversion)
        crsi_oversold = crsi[i] < 15  # Strong oversold
        crsi_overbought = crsi[i] > 85  # Strong overbought
        crsi_extreme_oversold = crsi[i] < 10
        crsi_extreme_overbought = crsi[i] > 90
        
        # Price position vs Bollinger Bands
        price_near_lower = close[i] < bb_lower[i] * 1.005  # Within 0.5% of lower band
        price_near_upper = close[i] > bb_upper[i] * 0.995  # Within 0.5% of upper band
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: CRSI extreme oversold + price near lower BB + 4h bull trend
        if crsi_extreme_oversold and price_near_lower and bull_trend:
            new_signal = SIZE_MAX
        # Secondary: CRSI oversold + range regime + price above 4h HMA
        elif crsi_oversold and range_regime and bull_trend:
            new_signal = SIZE_BASE
        # Tertiary: CRSI oversold + price near lower BB + EMA bullish
        elif crsi_oversold and price_near_lower and ema_bullish:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: CRSI extreme overbought + price near upper BB + 4h bear trend
        if crsi_extreme_overbought and price_near_upper and bear_trend:
            new_signal = -SIZE_MAX
        # Secondary: CRSI overbought + range regime + price below 4h HMA
        elif crsi_overbought and range_regime and bear_trend:
            new_signal = -SIZE_BASE
        # Tertiary: CRSI overbought + price near upper BB + EMA bearish
        elif crsi_overbought and price_near_upper and ema_bearish:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = highest_close - 2.0 * atr[i]
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
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = lowest_close + 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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