#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR breakout with 1d trend filter and volume confirmation.
# Long when price breaks above ATR(20) band + 1d close > SMA50 + volume > 20-period MA.
# Short when price breaks below ATR(20) band + 1d close < SMA50 + volume > 20-period MA.
# Uses ATR for volatility-adaptive breakouts, 1d trend for direction, volume for confirmation.
# Works in trending markets (breakouts) and avoids chop via trend filter.
# Target: 60-150 total trades over 4 years (15-38/year) with controlled risk.

name = "6h_atr_breakout_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d SMA50 for trend filter
    sma50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # ATR(20) for volatility band
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Upper and lower bands: close ± ATR * multiplier
    mult = 1.5
    upper_band = close + atr * mult
    lower_band = close - atr * mult
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(sma50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Exit: price closes below lower band (breakdown) or trend turns bearish
            if close[i] < lower_band[i] or sma50_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above upper band (breakout) or trend turns bullish
            if close[i] > upper_band[i] or sma50_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend filter
            if vol_filter:
                # Bullish breakout: price above upper band + 1d uptrend
                if close[i] > upper_band[i] and sma50_1d_aligned[i] < close[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price below lower band + 1d downtrend
                elif close[i] < lower_band[i] and sma50_1d_aligned[i] > close[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals