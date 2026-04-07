#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA200 filter + volume confirmation
# Long when Williams %R(14) crosses above -80 (oversold), price > 1d EMA200 (bullish regime), and volume > 1.5x 6h average volume
# Short when Williams %R(14) crosses below -20 (overbought), price < 1d EMA200 (bearish regime), and volume > 1.5x 6h average volume
# Exit when Williams %R crosses back above -20 (for longs) or below -80 (for shorts)
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Williams %R for mean reversion in trending markets, EMA200 for regime filter
# Target: 75-150 total trades over 4 years (19-38/year)

name = "6h_williamsr14_1d_ema200_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R(14) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    willr = willr.values
    
    # 1d EMA200 for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 6h volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(willr[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses back above -20 (exiting overbought)
            elif willr[i] > -20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses back below -80 (exiting oversold)
            elif willr[i] < -80:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Williams %R crossover, regime filter, and volume confirmation
            # Bullish crossover: Williams %R crosses above -80 (exiting oversold)
            bullish_cross = willr[i] > -80 and willr[i-1] <= -80
            # Bearish crossover: Williams %R crosses below -20 (entering overbought)
            bearish_cross = willr[i] < -20 and willr[i-1] >= -20
            
            # Long: bullish crossover, bullish regime (price > EMA200), volume spike
            if (bullish_cross and
                close[i] > ema200_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish crossover, bearish regime (price < EMA200), volume spike
            elif (bearish_cross and
                  close[i] < ema200_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals