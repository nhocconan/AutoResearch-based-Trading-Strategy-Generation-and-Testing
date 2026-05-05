#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility filter
# Long when price breaks above Donchian(20) high AND 1d close > 1d EMA50 AND ATR(14) < ATR(50) (low volatility regime)
# Short when price breaks below Donchian(20) low AND 1d close < 1d EMA50 AND ATR(14) < ATR(50)
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-40 trades/year per symbol.
# Donchian provides structural breakouts, 1d EMA50 filters for primary trend direction,
# ATR ratio ensures trades occur in low volatility environments reducing whipsaw.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Low volatility filter avoids false breakouts during high volatility chop.

name = "4h_Donchian20_1dEMA50_ATR_LowVol"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data ONCE before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian high: highest high over last 20 periods
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over last 20 periods
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Shift to use previous bar's levels (breakout of previous bar's Donchian)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    donchian_high[0] = np.nan  # First value invalid after roll
    donchian_low[0] = np.nan
    
    # Align Donchian levels to prices timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 4h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Calculate ATR(14) and ATR(50) for volatility filter on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First TR
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Low volatility filter: ATR(14) < ATR(50) indicates decreasing volatility
    low_volatility = atr_14 < atr_50
    
    # Align volatility filter to 4h timeframe
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(low_volatility_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian high AND 1d uptrend AND low volatility
            if (close[i] > donchian_high_aligned[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                low_volatility_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian low AND 1d downtrend AND low volatility
            elif (close[i] < donchian_low_aligned[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  low_volatility_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian low OR 1d trend changes to downtrend OR volatility increases
            if (close[i] < donchian_low_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5 or 
                low_volatility_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian high OR 1d trend changes to uptrend OR volatility increases
            if (close[i] > donchian_high_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5 or 
                low_volatility_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals