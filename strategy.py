# 4h_Triple_Confirmation_Breakout
# Hypothesis: 4h breakout with volume confirmation and trend strength filter
# Uses: Price channel breakout (ATR-based), volume surge (3x average), and ADX > 25
# Designed for 4h timeframe to target 20-50 trades/year (80-200 total over 4 years)
# Works in bull/bear via ADX trend filter and volatility-adjusted breakouts

name = "4h_Triple_Confirmation_Breakout"
timeframe = "4h"
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
    
    # ATR for volatility measurement (4h timeframe)
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i < period:
                if i == 0:
                    atr[i] = tr[0]
                else:
                    atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    # ADX for trend strength (4h timeframe)
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
            if len(data) < period:
                return result
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    # Calculate indicators on 4h data
    atr_4h = calculate_atr(high, low, close, 14)
    adx_4h = calculate_adx(high, low, close, 14)
    
    # Price channels based on ATR (dynamic support/resistance)
    upper_channel = np.full_like(close, np.nan)
    lower_channel = np.full_like(close, np.nan)
    
    for i in range(1, n):
        if not np.isnan(atr_4h[i-1]):
            upper_channel[i] = close[i-1] + (1.5 * atr_4h[i-1])
            lower_channel[i] = close[i-1] - (1.5 * atr_4h[i-1])
        else:
            upper_channel[i] = upper_channel[i-1] if i > 0 else close[i]
            lower_channel[i] = lower_channel[i-1] if i > 0 else close[i]
    
    # Volume confirmation: volume > 2.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_4h[i]) or np.isnan(adx_4h[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_4h[i] > 25
        
        if position == 0:
            # Long: price breaks above upper channel with volume and strong trend
            if (close[i] > upper_channel[i] and 
                volume_confirm[i] and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume and strong trend
            elif (close[i] < lower_channel[i] and 
                  volume_confirm[i] and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower channel or trend weakens (ADX < 20)
            if (close[i] < lower_channel[i]) or (adx_4h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper channel or trend weakens (ADX < 20)
            if (close[i] > upper_channel[i]) or (adx_4h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals