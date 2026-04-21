#!/usr/bin/env python3
"""
4h_HTF_1d_WilliamsFractal_TrendRegime_V1
Hypothesis: Use 1d Williams Fractals (bearish = resistance, bullish = support) with 4h EMA trend filter and volume confirmation. 
Enter short at bearish fractal breaks below in downtrend (EMA50 > EMA200), enter long at bullish fractal breaks above in uptrend (EMA50 < EMA200). 
ATR trailing stop (2.0x) to manage risk. Position size 0.25. Target 20-40 trades/year per symbol.
Works in both bull/bear by trend-filtering fractal breaks to trade with the higher-timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for 1d Williams Fractals
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # === 1d Williams Fractals ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA50 and EMA200 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume MA (20-period) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high = 0.0  # for long trailing stop
    lowest_low = 0.0    # for short trailing stop
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) 
            or np.isnan(ema50[i]) or np.isnan(ema200[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Trend filter: uptrend if EMA50 > EMA200, downtrend if EMA50 < EMA200
        uptrend = ema50[i] > ema200[i]
        downtrend = ema50[i] < ema200[i]
        
        if position == 0:
            # Long: bullish fractal break above in uptrend with volume
            if bullish_aligned[i-1] > 0 and price > bullish_aligned[i-1] and uptrend and vol_ok:
                signals[i] = 0.25
                position = 1
                highest_high = price
            # Short: bearish fractal break below in downtrend with volume
            elif bearish_aligned[i-1] > 0 and price < bearish_aligned[i-1] and downtrend and vol_ok:
                signals[i] = -0.25
                position = -1
                lowest_low = price
        
        elif position == 1:
            # Update highest high for trailing stop
            if price > highest_high:
                highest_high = price
            # ATR trailing stop: exit if price drops 2.0*ATR from highest high
            if price < highest_high - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low for trailing stop
            if price < lowest_low:
                lowest_low = price
            # ATR trailing stop: exit if price rises 2.0*ATR from lowest low
            if price > lowest_low + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_1d_WilliamsFractal_TrendRegime_V1"
timeframe = "4h"
leverage = 1.0