#!/usr/bin/env python3
"""
6h_EMA_Ribbon_RSI_Filter
Hypothesis: EMA ribbon (8/21/55) alignment with RSI(14) filter on 6h timeframe
- Bullish: EMA8 > EMA21 > EMA55 AND RSI between 40-60 (avoid overbought/oversold)
- Bearish: EMA8 < EMA21 < EMA55 AND RSI between 40-60
- Uses 1d ADX(14) > 20 to filter for trending markets, avoids chop
- Designed for low frequency (target 50-150 trades over 4 years) with clear trend signals
- Works in bull/bear via ADX filter and EMA ribbon direction
"""

name = "6h_EMA_Ribbon_RSI_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA calculation
    def calculate_ema(data, span):
        return pd.Series(data).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    # RSI calculation
    def calculate_rsi(data, period=14):
        delta = np.diff(data, prepend=data[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Wilder's smoothing
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                        result[i] = result[i-1] + alpha * (data[i] - result[i-1])
                    else:
                        result[i] = np.nan
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    # Calculate EMAs on 6h data
    ema8 = calculate_ema(close, 8)
    ema21 = calculate_ema(close, 21)
    ema55 = calculate_ema(close, 55)
    
    # Calculate RSI on 6h data
    rsi = calculate_rsi(close, 14)
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # Need enough data for EMA55
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(ema55[i]) or 
            np.isnan(rsi[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # EMA ribbon conditions
        ema_bullish = ema8[i] > ema21[i] > ema55[i]
        ema_bearish = ema8[i] < ema21[i] < ema55[i]
        
        # RSI filter: avoid extremes (40-60 range)
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # ADX filter: trending market
        trending = adx_1d_aligned[i] > 20
        
        if position == 0:
            # Long: bullish EMA ribbon + RSI neutral + trending
            if ema_bullish and rsi_neutral and trending:
                signals[i] = 0.25
                position = 1
            # Short: bearish EMA ribbon + RSI neutral + trending
            elif ema_bearish and rsi_neutral and trending:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if EMA ribbon breaks down or RSI goes overbought
            if not ema_bullish or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if EMA ribbon breaks up or RSI goes oversold
            if not ema_bearish or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals