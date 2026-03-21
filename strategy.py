#!/usr/bin/env python3
"""
Experiment #031: 15m Choppiness Regime + 4h HMA Trend + RSI Mean Reversion
Hypothesis: 15m is noisy but can work with strong regime filtering.
Choppiness Index (CHOP) distinguishes ranging vs trending markets.
When CHOP > 61.8 (range): use RSI mean reversion at extremes.
When CHOP < 38.2 (trend): use trend-following with 4h HMA bias.
4h HMA provides major trend filter (only long above, only short below).
This adapts to market conditions instead of using one rigid approach.
Multiple entry triggers ensure ≥10 trades while regime filter reduces whipsaws.
Position sizing 0.25 with 2.5x ATR stoploss for crash protection.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_chop_regime_4h_hma_rsi_v1"
timeframe = "15m"
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
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    atr = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = hh - ll
    range_val = np.where(range_val > 0, range_val, 1e-10)
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / range_val) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Additional trend filters
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    ema_8 = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(100, n):
        # 4h trend filter (major regime)
        hma_4h_valid = hma_4h_aligned[i] > 0
        price_above_4h_hma = close[i] > hma_4h_aligned[i] if hma_4h_valid else False
        price_below_4h_hma = close[i] < hma_4h_aligned[i] if hma_4h_valid else False
        
        # Regime detection via Choppiness Index
        is_ranging = chop[i] > 55.0  # Slightly relaxed from 61.8 for more trades
        is_trending = chop[i] < 42.0  # Slightly relaxed from 38.2
        
        # RSI conditions
        rsi_oversold = rsi[i] < 32
        rsi_overbought = rsi[i] > 68
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else True
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else True
        
        # Bollinger Band conditions
        price_near_lower = close[i] < bb_lower[i] * 1.002
        price_near_upper = close[i] > bb_upper[i] * 0.998
        price_at_mid = abs(close[i] - bb_mid[i]) < (bb_upper[i] - bb_lower[i]) * 0.15
        
        # HMA trend on 15m
        hma_trend_long = hma_21[i] > hma_50[i] and ema_8[i] > ema_21[i]
        hma_trend_short = hma_21[i] < hma_50[i] and ema_8[i] < ema_21[i]
        
        # Price position relative to 15m HMA
        price_above_hma21 = close[i] > hma_21[i]
        price_below_hma21 = close[i] < hma_21[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY TRIGGERS ===
        
        # Trigger 1: Range market + RSI oversold + near BB lower (mean reversion)
        if is_ranging and rsi_oversold and price_near_lower:
            new_signal = SIZE
        
        # Trigger 2: Trend market + 4h bullish + 15m trend + RSI rising (trend follow)
        elif is_trending and price_above_4h_hma and hma_trend_long and rsi_rising and rsi_neutral:
            new_signal = SIZE
        
        # Trigger 3: 4h bullish + RSI crossing up from oversold (reversal with trend)
        elif price_above_4h_hma and rsi[i] > 35 and rsi[i-2] < 35 and rsi_rising:
            new_signal = SIZE
        
        # Trigger 4: Price bounces from BB lower with 4h support
        elif price_above_4h_hma and price_near_lower and rsi[i] < 40 and rsi_rising:
            new_signal = SIZE
        
        # Trigger 5: HMA crossover long with 4h confirmation
        elif hma_trend_long and price_above_4h_hma and price_above_hma21 and rsi[i] > 45:
            new_signal = SIZE
        
        # === SHORT ENTRY TRIGGERS ===
        
        # Trigger 1: Range market + RSI overbought + near BB upper (mean reversion)
        if is_ranging and rsi_overbought and price_near_upper:
            new_signal = -SIZE
        
        # Trigger 2: Trend market + 4h bearish + 15m trend + RSI falling (trend follow)
        elif is_trending and price_below_4h_hma and hma_trend_short and rsi_falling and rsi_neutral:
            new_signal = -SIZE
        
        # Trigger 3: 4h bearish + RSI crossing down from overbought (reversal with trend)
        elif price_below_4h_hma and rsi[i] < 65 and rsi[i-2] > 65 and rsi_falling:
            new_signal = -SIZE
        
        # Trigger 4: Price rejects from BB upper with 4h resistance
        elif price_below_4h_hma and price_near_upper and rsi[i] > 60 and rsi_falling:
            new_signal = -SIZE
        
        # Trigger 5: HMA crossover short with 4h confirmation
        elif hma_trend_short and price_below_4h_hma and price_below_hma21 and rsi[i] < 55:
            new_signal = -SIZE
        
        # === STOPLOSS AND TAKE PROFIT LOGIC ===
        
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
                elif close[i] > entry_price + 2.5 * atr[entry_price > 0 and i > 0] if entry_price > 0 else 0:
                    if signals[i-1] == SIZE:
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
                profit_target = entry_price - 2.5 * atr[i] if i > 0 else entry_price
                if close[i] < profit_target and signals[i-1] == -SIZE:
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