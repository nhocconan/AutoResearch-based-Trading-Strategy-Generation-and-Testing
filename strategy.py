#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + RSI + Chop regime filter
# KAMA (Kaufman Adaptive Moving Average) adapts to market noise - follows trend in trending markets, stays flat in ranging markets
# RSI(14) for momentum confirmation - avoids counter-trend entries
# Chop index to filter regimes - only trade when Chop > 61.8 (ranging) for mean reversion or Chop < 38.2 (trending) for trend following
# Daily trend filter from 1d timeframe to align with higher timeframe bias
# Designed to reduce false signals and whipsaws in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Load 12h data for indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    dir = np.abs(np.diff(close, prepend=close[0]))
    for i in range(1, len(dir)):
        dir[i] = dir[i-1] + np.abs(close[i] - close[i-1])
    er = np.where(dir != 0, change / dir, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Chop Index (Choppiness Index)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max/Min range over period
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14), 
                    50)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend from daily timeframe
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Chop regime filters
        chop_high = chop[i] > 61.8  # Ranging market
        chop_low = chop[i] < 38.2   # Trending market
        
        price = close[i]
        
        if position == 0:
            # Enter long: KAMA turning up in uptrend OR ranging market with RSI oversold
            long_signal = False
            if vol_filter[i]:
                if is_uptrend and kama[i] > kama[i-1]:  # KAMA rising in uptrend
                    long_signal = True
                elif chop_high and rsi[i] < 30:  # RSI oversold in ranging market
                    long_signal = True
            
            # Enter short: KAMA turning down in downtrend OR ranging market with RSI overbought
            short_signal = False
            if vol_filter[i]:
                if is_downtrend and kama[i] < kama[i-1]:  # KAMA falling in downtrend
                    short_signal = True
                elif chop_high and rsi[i] > 70:  # RSI overbought in ranging market
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turning down OR RSI overbought
            exit_signal = False
            if kama[i] < kama[i-1] or rsi[i] > 70:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turning up OR RSI oversold
            exit_signal = False
            if kama[i] > kama[i-1] or rsi[i] < 30:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_ChopRegime"
timeframe = "12h"
leverage = 1.0