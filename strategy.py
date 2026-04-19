#!/usr/bin/env python3
"""
1h_RSI_MeanReversion_RangeFilter
Hypothesis: In 1h timeframe, use RSI(14) for mean reversion entries (RSI<30 long, RSI>70 short)
but only during ranging markets identified by ADX(14)<20 on 4h timeframe to avoid trending whipsaws.
Volume confirmation filters for institutional participation.
Designed for 1h to target 60-150 total trades over 4 years (15-37/year) with tight entries.
Works in bull/bear via range filter - avoids trending markets where mean reversion fails.
"""

name = "1h_RSI_MeanReversion_RangeFilter"
timeframe = "1h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # RSI(14) for mean reversion signals
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # ADX(14) for trend strength - calculated on 4h to filter ranging markets
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
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 4h data
    adx_4h = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # RSI on 1h data
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Range filter: only trade when ADX < 20 (ranging market)
        ranging_market = adx_4h_aligned[i] < 20
        
        if position == 0:
            # Long: RSI oversold (<30) in ranging market with volume
            if (rsi[i] < 30 and 
                ranging_market and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) in ranging market with volume
            elif (rsi[i] > 70 and 
                  ranging_market and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if RSI returns to neutral (40-60) or trend emerges (ADX >= 25)
            if (rsi[i] >= 40 and rsi[i] <= 60) or (adx_4h_aligned[i] >= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if RSI returns to neutral (40-60) or trend emerges (ADX >= 25)
            if (rsi[i] >= 40 and rsi[i] <= 60) or (adx_4h_aligned[i] >= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals