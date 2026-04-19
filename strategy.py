#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_1d_Trend_Signal"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on daily high/low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal: Bearish (sell) fractal = high with 2 lower highs on each side
    # Bullish (buy) fractal = low with 2 higher lows on each side
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Align fractals to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Daily EMA34 for trend filter (only needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6-period ATR for volatility filter
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_6 = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_6[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_34 = ema_34_1d_aligned[i]
        atr = atr_6[i]
        bearish_fract = bearish_fractal_aligned[i] > 0.5
        bullish_fract = bullish_fractal_aligned[i] > 0.5
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: bullish fractal + price above EMA34 + volume
            if bullish_fract and price > ema_34 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal + price below EMA34 + volume
            elif bearish_fract and price < ema_34 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: bearish fractal or price below EMA34
            if bearish_fract or price < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bullish fractal or price above EMA34
            if bullish_fract or price > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals