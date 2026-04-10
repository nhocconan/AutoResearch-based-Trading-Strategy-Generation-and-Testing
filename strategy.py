#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h ADX regime filter
# - Long when Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (strong trend)
# - Short when Bear Power < 0 AND Bull Power > 0 AND 12h ADX > 25 (strong trend)
# - Exit when Bull Power and Bear Power converge (both near zero) OR ADX < 20 (weak trend)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear power relative to EMA13, effective in trending markets
# - ADX filter ensures we only trade when trend is strong enough to sustain moves
# - Works in both bull (long bias) and bear (short bias) markets via ADX regime

name = "6h_12h_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Pre-compute 12h ADX(14) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_12h = np.zeros_like(tr)
    atr_12h[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and ATR
    def WilderSmooth(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.mean(arr[1:period+1])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr14 = WilderSmooth(tr, 14)
    plus_dm14 = WilderSmooth(plus_dm, 14)
    minus_dm14 = WilderSmooth(minus_dm, 14)
    
    # DI and ADX
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    dx = np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100
    dx[~np.isfinite(dx)] = 0
    
    adx_12h = WilderSmooth(dx, 14)
    
    # Align HTF indicators to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx_12h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND strong trend (ADX > 25)
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx_12h_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power < 0 AND Bull Power > 0 AND strong trend (ADX > 25)
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  adx_12h_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Power convergence OR weak trend (ADX < 20)
            power_convergence = (abs(bull_power[i]) < 0.1 * close[i] and 
                                abs(bear_power[i]) < 0.1 * close[i])
            weak_trend = adx_12h_aligned[i] < 20
            
            if power_convergence or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals