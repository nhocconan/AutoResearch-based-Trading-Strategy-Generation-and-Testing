#!/usr/bin/env python3
"""
Experiment #011: 12h KAMA Adaptive Trend + Daily Regime Filter + BBW Volatility
Hypothesis: 12h timeframe balances noise reduction with trade frequency.
KAMA (Kaufman Adaptive MA) adapts to market efficiency - fast in trends, slow in ranges.
Daily HMA provides major trend regime (bull/bear).
Bollinger BandWidth percentile detects squeeze (mean reversion) vs expansion (trend).
RSI confirms momentum without being overbought/oversold.
ATR-based stoploss at 2.5x protects against crashes like 2022.
Position sizing: 0.25 full, 0.125 partial - discrete levels to minimize fee churn.
This should work across BTC/ETH/SOL with adaptive behavior per regime.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_bbw_regime_12h_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    volatility[0] = change[0]
    er = np.where(volatility > 0, change / volatility, 0.0)
    # Calculate smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    # Calculate KAMA
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and BandWidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = np.where(sma > 0, (upper - lower) / sma, 0.0)
    return upper, lower, bandwidth

def calculate_bbwidth_percentile(bandwidth, lookback=100):
    """Calculate Bollinger BandWidth percentile rank."""
    bw_s = pd.Series(bandwidth)
    percentile = bw_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: np.sum(x < x.iloc[-1]) / len(x), raw=False
    ).values
    return np.nan_to_num(percentile, nan=0.5)

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
    kama_fast = calculate_kama(close, period=10, fast=2, slow=30)
    kama_slow = calculate_kama(close, period=20, fast=2, slow=30)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bbw_pct = calculate_bbwidth_percentile(bb_width, 100)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=0.0)
    
    signals = np.zeros(n)
    SIZE_FULL = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    highest_price = np.zeros(n)
    lowest_price = np.zeros(n)
    
    for i in range(100, n):
        # Daily trend regime (major direction)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # KAMA crossover signals (adaptive trend)
        kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        kama_trend_long = kama_fast[i] > kama_slow[i]
        kama_trend_short = kama_fast[i] < kama_slow[i]
        
        # RSI momentum (not extreme)
        rsi_long = rsi[i] > 40 and rsi[i] < 75
        rsi_short = rsi[i] < 60 and rsi[i] > 25
        rsi_rising = rsi[i] > rsi[i-5] if i > 5 else True
        rsi_falling = rsi[i] < rsi[i-5] if i > 5 else True
        
        # Bollinger BandWidth regime
        bbw_squeeze = bbw_pct[i] < 0.30  # Low volatility = mean reversion regime
        bbw_expand = bbw_pct[i] > 0.50   # High volatility = trend regime
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        # Price position in BB
        price_mid = (bb_upper[i] + bb_lower[i]) / 2
        price_lower_third = close[i] < price_mid
        price_upper_third = close[i] > price_mid
        
        # Entry logic - relaxed for sufficient trades
        new_signal = 0.0
        
        # LONG entries (multiple paths to ensure trades)
        # Path 1: Daily bullish + KAMA trend + RSI ok + volume
        if daily_bullish and kama_trend_long and rsi_long and vol_confirm:
            new_signal = SIZE_FULL
        # Path 2: KAMA crossover with daily support
        elif daily_bullish and kama_cross_long and rsi[i] > 35:
            new_signal = SIZE_FULL
        # Path 3: Mean reversion in squeeze (buy lower band)
        elif bbw_squeeze and price_lower_third and rsi[i] < 50 and daily_bullish:
            new_signal = SIZE_FULL
        # Path 4: Trend continuation (less strict)
        elif kama_trend_long and rsi_rising and close[i] > kama_slow[i]:
            new_signal = SIZE_FULL
        
        # SHORT entries (more conservative - asymmetric)
        # Path 1: Daily bearish + KAMA trend + RSI ok
        if daily_bearish and kama_trend_short and rsi_short and vol_confirm:
            new_signal = -SIZE_FULL
        # Path 2: KAMA crossover with daily resistance
        elif daily_bearish and kama_cross_short and rsi[i] < 65:
            new_signal = -SIZE_FULL
        # Path 3: Mean reversion in squeeze (sell upper band)
        elif bbw_squeeze and price_upper_third and rsi[i] > 50 and daily_bearish:
            new_signal = -SIZE_FULL
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs - take partial profit at 3R
            elif close[i] > entry_price[i-1] + 3.0 * atr[i]:
                if new_signal == SIZE_FULL:
                    new_signal = SIZE_HALF  # Reduce to half
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts - take partial profit at 3R
            elif close[i] < entry_price[i-1] - 3.0 * atr[i]:
                if new_signal == -SIZE_FULL:
                    new_signal = -SIZE_HALF  # Reduce to half
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            highest_price[i] = close[i]
            lowest_price[i] = close[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                highest_price[i] = close[i]
                lowest_price[i] = close[i]
            else:
                highest_price[i] = max(highest_price[i-1], close[i])
                lowest_price[i] = min(lowest_price[i-1], close[i])
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            highest_price[i] = highest_price[i-1] if i > 0 else close[i]
            lowest_price[i] = lowest_price[i-1] if i > 0 else close[i]
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals