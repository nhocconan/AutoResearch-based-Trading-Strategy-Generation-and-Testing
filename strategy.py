#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index (CHOP) for regime filter.
- Long when: price > KAMA(14, ER=10) AND RSI > 50 AND CHOP(14) < 61.8 (trending market)
- Short when: price < KAMA(14, ER=10) AND RSI < 50 AND CHOP(14) < 61.8 (trending market)
- Exit when: price crosses KAMA in opposite direction OR RSI reaches extreme (70/30) OR market becomes choppy (CHOP > 61.8)
- Position size: 0.25. Target: 30-100 trades over 4 years (7-25/year).
- Works in both bull and bear: KAMA adapts to volatility, CHOP filter avoids whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close
    close_1d = df_1d['close'].values
    kama_1d = calculate_kama(close_1d, er_period=10, fast_ema=2, slow_ema=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) on 1d close
    rsi_1d = calculate_rsi(close_1d, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Choppiness Index on 1d OHLC
    chop_1d = calculate_choppiness_index(df_1d['high'].values, df_1d['low'].values, close_1d, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (30) and RSI/CHOP (14)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend direction from KAMA
        price_above_kama = close[i] > kama_1d_aligned[i]
        price_below_kama = close[i] < kama_1d_aligned[i]
        
        # Determine market regime from Choppiness Index
        trending_market = chop_1d_aligned[i] < 61.8  # CHOP < 61.8 = trending
        choppy_market = chop_1d_aligned[i] >= 61.8   # CHOP >= 61.8 = ranging/choppy
        
        if position == 0:
            if trending_market:
                # Trending market: follow KAMA direction with RSI confirmation
                long_setup = price_above_kama and (rsi_1d_aligned[i] > 50)
                short_setup = price_below_kama and (rsi_1d_aligned[i] < 50)
            else:
                # Choppy market: no new entries (avoid whipsaws)
                long_setup = False
                short_setup = False
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            exit_signal = (
                not price_above_kama or  # price crossed below KAMA
                rsi_1d_aligned[i] >= 70 or  # RSI overbought
                choppy_market  # market became choppy
            )
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            exit_signal = (
                not price_below_kama or  # price crossed above KAMA
                rsi_1d_aligned[i] <= 30 or  # RSI oversold
                choppy_market  # market became choppy
            )
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

def calculate_kama(close, er_period=10, fast_ema=2, slow_ema=30):
    """Kaufman Adaptive Moving Average"""
    close = pd.Series(close)
    direction = np.abs(close.diff(er_period))
    volatility = close.diff().abs().rolling(window=er_period, min_periods=1).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = [np.nan] * len(close)
    kama[er_period] = close.iloc[er_period]
    for i in range(er_period+1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close.iloc[i] - kama[i-1])
    return np.array(kama)

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    close = pd.Series(close)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.replace([np.inf, -np.inf], 50).fillna(50).values

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0