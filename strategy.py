#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and ATR(14) volatility filter
# Long when price breaks above Donchian upper band AND 1d close > 1d EMA34 AND ATR(14) < 0.04 * close
# Short when price breaks below Donchian lower band AND 1d close < 1d EMA34 AND ATR(14) < 0.04 * close
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-25 trades/year per symbol.
# Donchian provides structure; EMA34 filters trend; ATR filter avoids high volatility chop.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# 12h timeframe reduces trade frequency to minimize fee drag while capturing medium-term trends.

name = "12h_Donchian20_1dEMA34_ATR_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian bands (20-period) based on previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper band: highest high of previous 20 bars
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low of previous 20 bars
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Shift to use previous bar's levels (breakout of previous bar's Donchian)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    donchian_high[0] = np.nan  # First value invalid after roll
    donchian_low[0] = np.nan
    
    # Align Donchian bands to prices timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 12h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Volatility filter: ATR < 4% of price (avoid high volatility chop)
    vol_filter = atr_14 < (0.04 * close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian upper AND 1d uptrend AND low volatility
            if (close[i] > donchian_high_aligned[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian lower AND 1d downtrend AND low volatility
            elif (close[i] < donchian_low_aligned[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian lower OR 1d trend changes to downtrend
            if (close[i] < donchian_low_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian upper OR 1d trend changes to uptrend
            if (close[i] > donchian_high_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals