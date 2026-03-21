#!/usr/bin/env python3
"""
Experiment #028: 4h Mean Reversion with Daily Trend + Choppiness Regime Filter
Hypothesis: 4h timeframe captures multi-day swings while avoiding intraday noise.
Daily HMA provides major trend regime (only trade mean reversion WITH the trend).
Choppiness Index (CHOP) detects range vs trend regimes - mean revert when CHOP>61.8.
4h RSI extremes (20/80) + Bollinger position provide entry triggers.
This should work better in 2022 crash and 2025 bear market than pure trend following.
Multiple entry triggers ensure ≥10 trades per symbol. Position size 0.30 with 2.5x ATR stop.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_mr_daily_chop_rsi_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    # Position within bands: 0=lower, 0.5=middle, 1=upper
    bb_pos = np.zeros(len(close))
    band_width = upper - lower
    valid = band_width > 0
    bb_pos[valid] = (close[valid] - lower[valid]) / band_width[valid]
    bb_pos = np.clip(bb_pos, 0, 1)
    return sma, upper, lower, bb_pos

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    valid = (price_range > 0) & (atr_sum > 0)
    
    chop = np.zeros(len(close))
    chop[valid] = 100 * np.log10(atr_sum[valid] / price_range[valid]) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Helps identify reversal points in trending markets.
    """
    hl2 = (high + low) / 2
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    valid = price_range > 0
    
    xfm = np.zeros(len(hl2))
    xfm[valid] = 0.66 * ((hl2[valid] - lowest[valid]) / price_range[valid] - 0.5) + 0.67 * np.roll(xfm, 1)[valid]
    xfm = np.clip(xfm, -0.99, 0.99)
    
    fisher = 0.5 * np.log((1 + xfm) / (1 - xfm))
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    return fisher, fisher_prev

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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for entries
    sma_20, bb_upper, bb_lower, bb_pos = calculate_bollinger(close, 20, 2.0)
    chop = calculate_choppiness(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, 9)
    
    # Additional trend filters
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    max_profit = 0.0
    
    for i in range(100, n):
        # Daily trend regime filter
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # Choppiness regime (mean reversion works when CHOP > 61.8)
        is_ranging = chop[i] > 55  # Slightly relaxed for more trades
        is_trending = chop[i] < 45
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_very_oversold = rsi_7[i] < 25
        rsi_very_overbought = rsi_7[i] > 75
        
        # Bollinger Band position
        bb_near_lower = bb_pos[i] < 0.15
        bb_near_upper = bb_pos[i] > 0.85
        
        # Fisher Transform reversals
        fisher_long = fisher[i] > fisher_prev[i] and fisher_prev[i] < -1.0
        fisher_short = fisher[i] < fisher_prev[i] and fisher_prev[i] > 1.0
        
        # HMA trend on 4h
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # Entry logic - MULTIPLE triggers to ensure trades (Rule 9)
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS
        # Trigger 1: RSI oversold + daily bullish (mean reversion with trend)
        if rsi_oversold and daily_bullish:
            new_signal = SIZE
        # Trigger 2: RSI very oversold + ranging market (pure mean reversion)
        elif rsi_very_oversold and is_ranging:
            new_signal = SIZE
        # Trigger 3: Bollinger lower + RSI ok + daily support
        elif bb_near_lower and rsi[i] < 50 and daily_bullish:
            new_signal = SIZE
        # Trigger 4: Fisher reversal long + HMA trend
        elif fisher_long and hma_trend_long:
            new_signal = SIZE
        # Trigger 5: RSI rising from oversold + price above HMA21
        elif rsi_7[i] > rsi_7[i-3] and rsi_7[i-3] < 30 and close[i] > hma_21[i]:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS
        # Trigger 1: RSI overbought + daily bearish (mean reversion with trend)
        if rsi_overbought and daily_bearish:
            new_signal = -SIZE
        # Trigger 2: RSI very overbought + ranging market (pure mean reversion)
        elif rsi_very_overbought and is_ranging:
            new_signal = -SIZE
        # Trigger 3: Bollinger upper + RSI ok + daily resistance
        elif bb_near_upper and rsi[i] > 50 and daily_bearish:
            new_signal = -SIZE
        # Trigger 4: Fisher reversal short + HMA trend
        elif fisher_short and hma_trend_short:
            new_signal = -SIZE
        # Trigger 5: RSI falling from overbought + price below HMA21
        elif rsi_7[i] < rsi_7[i-3] and rsi_7[i-3] > 70 and close[i] < hma_21[i]:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > entry_price:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] > entry_price + 2.5 * atr[entry_idx if 'entry_idx' in dir() else i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < entry_price:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] < entry_price - 2.5 * atr[entry_idx if 'entry_idx' in dir() else i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals