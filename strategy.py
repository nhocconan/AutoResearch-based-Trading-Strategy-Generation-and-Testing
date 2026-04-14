#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with daily ATR filter and volume confirmation
# Williams Fractals identify significant pivot points where price reverses.
# Breakouts above/below recent fractals with volume confirmation capture momentum.
# Daily ATR filter avoids trading in low volatility environments.
# Works in bull/bear by requiring breakouts in direction of daily trend (EMA50)
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for fractals, trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily ATR(14) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = high_low[0]  # first value
    low_close[0] = high_low[0]   # first value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Williams Fractals (5-bar: 2 left, 2 right)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] < low[n+2]
    n1 = len(high_1d)
    bearish_fractal = np.full(n1, np.nan)
    bullish_fractal = np.full(n1, np.nan)
    
    for i in range(2, n1 - 2):
        # Bearish fractal (peak)
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i+1] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal (trough)
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i+1] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals need 2 extra bars for confirmation (the 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: volume > 1.3x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        atr = atr_14_1d_aligned[i]
        
        # Skip if ATR is too low (avoid choppy markets)
        if atr < 0.0001 * price:  # essentially zero ATR
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above bearish fractal (resistance) with volume filter AND above daily EMA50
            if (not np.isnan(bearish_fractal_aligned[i]) and 
                price > bearish_fractal_aligned[i] and 
                price > ema_50_1d_aligned[i] and 
                vol > 1.3 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below bullish fractal (support) with volume filter AND below daily EMA50
            elif (not np.isnan(bullish_fractal_aligned[i]) and 
                  price < bullish_fractal_aligned[i] and 
                  price < ema_50_1d_aligned[i] and 
                  vol > 1.3 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below bullish fractal (support) OR below daily EMA50
            if (not np.isnan(bullish_fractal_aligned[i]) and price < bullish_fractal_aligned[i]) or \
               price < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above bearish fractal (resistance) OR above daily EMA50
            if (not np.isnan(bearish_fractal_aligned[i]) and price > bearish_fractal_aligned[i]) or \
               price > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Williams_Fractal_Breakout_EMA_Volume"
timeframe = "6h"
leverage = 1.0