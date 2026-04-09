#!/usr/bin/env python3
# 1d_keltner_breakout_volume_v1
# Hypothesis: 1d strategy using weekly EMA200 for trend, daily Keltner breakout with volume confirmation.
# Long when price breaks above daily Keltner upper band (EMA20 + 2*ATR) with volume > 1.5x 20-day average,
# price > weekly EMA200, and weekly ATR ratio < 0.8 (low volatility regime).
# Short when price breaks below daily Keltner lower band (EMA20 - 2*ATR) with volume > 1.5x 20-day average,
# price < weekly EMA200, and weekly ATR ratio < 0.8.
# Exit when price crosses back below/above daily EMA20.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 15-25 trades/year (60-100 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily EMA20 for Keltner center and exit
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Daily ATR(14) for Keltner bands
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Daily Keltner Bands
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr
    
    # Get weekly data for trend and volatility regime (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema200_1w = close_1w_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Weekly ATR(14) for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    tr1_1w = pd.Series(high_1w - low_1w)
    tr2_1w = pd.Series(np.abs(high_1w - np.roll(close_1w_arr, 1)))
    tr3_1w = pd.Series(np.abs(low_1w - np.roll(close_1w_arr, 1)))
    tr_1w = pd.concat([tr1_1w, tr2_1w, tr3_1w], axis=1).max(axis=1)
    atr_1w = tr_1w.rolling(window=14, min_periods=14).mean().values
    
    # Weekly ATR ratio (current ATR / 50-period average) for volatility filter
    atr_1w_s = pd.Series(atr_1w)
    atr_ma_50 = atr_1w_s.rolling(window=50, min_periods=50).mean().values
    atr_ratio_1w = atr_1w / atr_ma_50  # < 0.8 = low volatility regime
    
    # Align weekly data to daily timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    atr_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema20[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(atr_ratio_1w_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Trend filter: price > weekly EMA200 for long, price < weekly EMA200 for short
        # Volatility regime filter: weekly ATR ratio < 0.8 (low volatility)
        low_volatility = atr_ratio_1w_aligned[i] < 0.8
        
        if position == 1:  # Long position
            # Exit: Price crosses back below daily EMA20
            if close[i] < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses back above daily EMA20
            if close[i] > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume, trend, and volatility confirmation
            bullish_breakout = (close[i] > keltner_upper[i]) and volume_confirmed and (close[i] > ema200_1w_aligned[i]) and low_volatility
            bearish_breakout = (close[i] < keltner_lower[i]) and volume_confirmed and (close[i] < ema200_1w_aligned[i]) and low_volatility
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals