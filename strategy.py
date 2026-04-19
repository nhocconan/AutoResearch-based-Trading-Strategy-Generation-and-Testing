#!/usr/bin/env python3
# 12h_Keltner_Breakout_Volume_ADX_Filter
# Hypothesis: 12h Keltner Channel breakout with volume confirmation and ADX trend filter
# Keltner Channels use ATR to set dynamic bands, adapting to volatility
# Breakouts above upper band (bullish) or below lower band (bearish) with volume and ADX > 25 capture institutional moves
# Works in bull/bear via ADX filter and volatility-adjusted bands
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Keltner_Breakout_Volume_ADX_Filter"
timeframe = "12h"
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
    
    # ATR(10) for Keltner Channels and volatility filter
    def calculate_atr(high, low, close, period=10):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = np.zeros_like(close)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    # ADX(14) for trend strength filter
    def calculate_adx(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
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
    
    # Use 12h data for ATR and ADX calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ATR and ADX on 12h data
    atr_12h = calculate_atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 10)
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # EMA(20) for Keltner Channel middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands: EMA(20) ± 2 * ATR(10)
    upper_keltner = ema_20 + 2 * atr_12h_aligned
    lower_keltner = ema_20 - 2 * atr_12h_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(ema_20[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_12h_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above upper Keltner band with volume and strong trend
            if (close[i] > upper_keltner[i] and 
                volume_confirm[i] and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner band with volume and strong trend
            elif (close[i] < lower_keltner[i] and 
                  volume_confirm[i] and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Keltner band or trend weakens (ADX < 20)
            if (close[i] < lower_keltner[i]) or (adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Keltner band or trend weakens (ADX < 20)
            if (close[i] > upper_keltner[i]) or (adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals