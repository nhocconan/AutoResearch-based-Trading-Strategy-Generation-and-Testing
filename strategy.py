#!/usr/bin/env python3
"""
Experiment #020: 30m Regime-Adaptive Strategy with 4h Trend Filter
Hypothesis: 30m timeframe balances noise reduction with trade frequency.
Use Choppiness Index to detect regime (ranging vs trending).
In ranging markets (CHOP>61.8): mean reversion with RSI extremes.
In trending markets (CHOP<38.2): trend following with HMA alignment.
4h HMA provides major trend filter for directional bias.
ATR-based stoploss (2.5x) protects against adverse moves.
Position sizing: 0.25 discrete levels to minimize fee churn.
This should work in both 2021-2024 bull/bear cycles and 2025 bear market.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_chop_rsi_4h_v2"
timeframe = "30m"
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
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    atr = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # Choppiness Index
    chop = np.zeros(len(close))
    mask = price_range > 0
    chop[mask] = 100 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion detection."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = np.zeros(len(close))
    mask = std > 0
    zscore[mask] = (close[mask] - sma[mask]) / std[mask]
    return zscore

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    hma_fast = calculate_hma(close, 16)
    hma_slow = calculate_hma(close, 48)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.mean(volume[50:100]) if len(volume) > 100 else 1.0)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    
    for i in range(100, n):
        # 4h trend filter (major regime)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 30m HMA trend
        hma_trend_long = hma_fast[i] > hma_slow[i]
        hma_trend_short = hma_fast[i] < hma_slow[i]
        
        # Regime detection via Choppiness Index
        is_ranging = chop[i] > 55  # Slightly relaxed from 61.8 for more signals
        is_trending = chop[i] < 45  # Slightly relaxed from 38.2
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi[i] < 32
        rsi_overbought = rsi[i] > 68
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        # Z-score extremes
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # HMA crossover signals
        hma_cross_long = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_cross_short = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        new_signal = 0.0
        
        # === RANGING MARKET: Mean Reversion Strategy ===
        if is_ranging:
            # Long: RSI oversold + zscore oversold + 4h not strongly bearish
            if rsi_oversold and zscore_oversold and not trend_4h_bearish:
                new_signal = SIZE
            # Short: RSI overbought + zscore overbought + 4h not strongly bullish
            elif rsi_overbought and zscore_overbought and not trend_4h_bullish:
                new_signal = -SIZE
            # Exit mean reversion when RSI returns to neutral
            elif position_side > 0 and rsi_neutral:
                new_signal = 0.0
            elif position_side < 0 and rsi_neutral:
                new_signal = 0.0
        
        # === TRENDING MARKET: Trend Following Strategy ===
        elif is_trending:
            # Long: 4h bullish + 30m HMA trend + HMA crossover or pullback
            if trend_4h_bullish and hma_trend_long:
                if hma_cross_long and vol_confirm:
                    new_signal = SIZE
                elif rsi[i] > 45 and rsi[i] < 60 and rsi[i] > rsi[i-3]:
                    new_signal = SIZE
            # Short: 4h bearish + 30m HMA trend + HMA crossover or rally
            elif trend_4h_bearish and hma_trend_short:
                if hma_cross_short and vol_confirm:
                    new_signal = -SIZE
                elif rsi[i] > 40 and rsi[i] < 55 and rsi[i] < rsi[i-3]:
                    new_signal = -SIZE
        
        # === DEFAULT: HMA crossover with 4h filter (always active) ===
        if new_signal == 0.0:
            # Long on HMA cross with 4h support
            if hma_cross_long and (trend_4h_bullish or not trend_4h_bearish) and vol_confirm:
                new_signal = SIZE
            # Short on HMA cross with 4h resistance
            elif hma_cross_short and (trend_4h_bearish or not trend_4h_bullish) and vol_confirm:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs
            else:
                current_trail = close[i] - 2.5 * atr[i]
                if current_trail > trailing_stop[i-1] if i > 0 else 0:
                    trailing_stop[i] = current_trail
                else:
                    trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] > entry_price[i-1] + 2.5 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts
            else:
                current_trail = close[i] + 2.5 * atr[i]
                if current_trail < trailing_stop[i-1] if i > 0 else 999999:
                    trailing_stop[i] = current_trail
                else:
                    trailing_stop[i] = trailing_stop[i-1] if i > 0 else 999999
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] < entry_price[i-1] - 2.5 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0
        
        signals[i] = new_signal
    
    return signals