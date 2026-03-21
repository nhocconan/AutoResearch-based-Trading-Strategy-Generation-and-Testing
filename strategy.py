#!/usr/bin/env python3
"""
Experiment #017: 12h Regime-Adaptive Strategy with Choppiness Index
Hypothesis: 12h timeframe is ideal for multi-day swings. Use Choppiness Index to detect
regime (range vs trend), then apply appropriate strategy:
- Range (CHOP > 61.8): Mean reversion with RSI extremes + Bollinger bands
- Trend (CHOP < 38.2): Trend following with Supertrend + HMA confirmation
Daily HTF provides major trend bias filter. ATR stoploss at 2.5x protects capital.
Position sizing: 0.25 discrete levels to minimize fee churn while capturing moves.
This differs from previous 12h attempts by using regime detection instead of pure trend.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_chop_rsi_v1"
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
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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
    CHOP > 61.8 = range/choppy market (mean reversion favorable)
    CHOP < 38.2 = trending market (trend following favorable)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = hh - ll
    range_val = np.where(range_val <= 0, 1e-10, range_val)
    
    chop = 100 * np.log10(atr_sum / range_val) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform for reversal detection.
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    hl2 = (high + low) / 2
    
    # Normalize price to -1 to +1 range
    hh = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    range_val = hh - ll
    range_val = np.where(range_val <= 0, 1e-10, range_val)
    
    normalized = 2 * (hl2 - ll) / range_val - 1
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    
    # HMA for trend confirmation
    hma_fast = calculate_hma(close, 16)
    hma_slow = calculate_hma(close, 48)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=1.0)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative position size for 12h timeframe
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    trailing_stop = np.zeros(n)
    
    for i in range(100, n):
        # Daily trend bias (major regime filter)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # Regime detection via Choppiness Index
        is_range = chop[i] > 55.0  # Slightly relaxed from 61.8 for more signals
        is_trend = chop[i] < 45.0  # Slightly relaxed from 38.2
        
        # Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # HMA trend
        hma_bullish = hma_fast[i] > hma_slow[i]
        hma_bearish = hma_fast[i] < hma_slow[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        # Bollinger Band conditions
        bb_low = close[i] < bb_lower[i]
        bb_high = close[i] > bb_upper[i]
        
        # Fisher Transform signals
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        new_signal = 0.0
        
        # === RANGE MARKET STRATEGY (Mean Reversion) ===
        if is_range:
            # Long: RSI oversold + near BB lower + daily bullish bias preferred
            if rsi_oversold and (bb_low or close[i] < bb_mid):
                if daily_bullish or not daily_bearish:
                    new_signal = SIZE
            # Short: RSI overbought + near BB upper + daily bearish bias preferred
            elif rsi_overbought and (bb_high or close[i] > bb_mid):
                if daily_bearish or not daily_bullish:
                    new_signal = -SIZE
            # Fisher reversal in range
            elif fisher_long and rsi[i] < 50:
                new_signal = SIZE
            elif fisher_short and rsi[i] > 50:
                new_signal = -SIZE
        
        # === TREND MARKET STRATEGY (Trend Following) ===
        elif is_trend:
            # Long: Supertrend bullish + HMA bullish + daily bias aligned
            if st_long and hma_bullish:
                if daily_bullish or not daily_bearish:
                    new_signal = SIZE
            # Short: Supertrend bearish + HMA bearish + daily bias aligned
            elif st_short and hma_bearish:
                if daily_bearish or not daily_bullish:
                    new_signal = -SIZE
            # Supertrend flip entry
            elif st_direction[i] == 1 and st_direction[i-1] == -1 and vol_confirm:
                if daily_bullish or not daily_bearish:
                    new_signal = SIZE
            elif st_direction[i] == -1 and st_direction[i-1] == 1 and vol_confirm:
                if daily_bearish or not daily_bullish:
                    new_signal = -SIZE
        
        # === NEUTRAL/TRANSITION MARKET ===
        else:
            # Use Fisher Transform for reversals in uncertain markets
            if fisher_long and rsi[i] < 45:
                new_signal = SIZE
            elif fisher_short and rsi[i] > 55:
                new_signal = -SIZE
        
        # === STOPLOSS AND POSITION MANAGEMENT ===
        prev_side = position_side[i-1] if i > 0 else 0
        prev_entry = entry_price[i-1] if i > 0 else 0
        
        # Check stoploss for existing positions
        if prev_side > 0 and prev_entry > 0:
            stop_loss = prev_entry - 2.5 * atr[i]
            trail_stop = trailing_stop[i-1] if i > 0 else stop_loss
            trail_stop = max(trail_stop, prev_entry + 1.0 * atr[i])  # Trail up only
            
            if close[i] < stop_loss or close[i] < trail_stop:
                new_signal = 0.0  # Stoploss hit
            
            # Take partial profit at 3R
            if new_signal == SIZE and close[i] > prev_entry + 3.0 * atr[i]:
                new_signal = HALF_SIZE
        
        if prev_side < 0 and prev_entry > 0:
            stop_loss = prev_entry + 2.5 * atr[i]
            trail_stop = trailing_stop[i-1] if i > 0 else stop_loss
            trail_stop = min(trail_stop, prev_entry - 1.0 * atr[i])  # Trail down only
            
            if close[i] > stop_loss or close[i] > trail_stop:
                new_signal = 0.0  # Stoploss hit
            
            # Take partial profit at 3R
            if new_signal == -SIZE and close[i] < prev_entry - 3.0 * atr[i]:
                new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0:
            if prev_side == 0 or np.sign(new_signal) != prev_side:
                position_side[i] = np.sign(new_signal)
                entry_price[i] = close[i]
                if position_side[i] > 0:
                    trailing_stop[i] = close[i] - 2.5 * atr[i]
                else:
                    trailing_stop[i] = close[i] + 2.5 * atr[i]
            else:
                position_side[i] = prev_side
                entry_price[i] = prev_entry
                if position_side[i] > 0:
                    trailing_stop[i] = max(trailing_stop[i-1] if i > 0 else 0, close[i] - 2.5 * atr[i])
                else:
                    trailing_stop[i] = min(trailing_stop[i-1] if i > 0 else 999999, close[i] + 2.5 * atr[i])
        else:
            position_side[i] = 0
            entry_price[i] = prev_entry
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
        
        signals[i] = new_signal
    
    return signals