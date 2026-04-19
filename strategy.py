#!/usr/bin/env python3
# 12h_RSI_Overbought_Oversold_With_Volume_and_Trend_Filter
# Hypothesis: RSI extremes (below 30 for long, above 70 for short) on 12h timeframe with volume confirmation and ADX trend filter.
# Works in bull/bear markets: RSI mean reversion is effective in ranging markets, while ADX filter avoids false signals in strong trends.
# Volume confirmation ensures institutional participation. Targets 50-150 trades over 4 years.

name = "12h_RSI_Overbought_Oversold_With_Volume_and_Trend_Filter"
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
    
    # RSI(14) calculation
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # ADX(14) for trend strength filter
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
        
        # Avoid division by zero
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    # Get 12h data for RSI and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate RSI and ADX on 12h data
    rsi_12h = calculate_rsi(df_12h['close'].values, 14)
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # RSI thresholds: oversold <30, overbought >70
        rsi_oversold = rsi_12h_aligned[i] < 30
        rsi_overbought = rsi_12h_aligned[i] > 70
        
        # ADX filter: only trade when ADX > 20 (avoid extremely weak trends)
        not_weak_trend = adx_12h_aligned[i] > 20
        
        if position == 0:
            # Long: RSI oversold with volume and not weak trend
            if (rsi_oversold and 
                volume_confirm[i] and 
                not_weak_trend):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought with volume and not weak trend
            elif (rsi_overbought and 
                  volume_confirm[i] and 
                  not_weak_trend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI reaches overbought (>70) or trend weakens (ADX < 15)
            if (rsi_12h_aligned[i] > 70) or (adx_12h_aligned[i] < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI reaches oversold (<30) or trend weakens (ADX < 15)
            if (rsi_12h_aligned[i] < 30) or (adx_12h_aligned[i] < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals