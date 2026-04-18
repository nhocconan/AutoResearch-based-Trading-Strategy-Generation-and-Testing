#!/usr/bin/env python3
"""
1d_WideBand_Keltner_Breakout_Volume_Trend
Hypothesis: In volatile crypto markets, wide-band Keltner channels (ATR multiplier 2.5) capture major breakouts better than Bollinger bands. 
Breakouts above upper band with volume >1.5x average and price above weekly EMA50 capture institutional momentum.
Exit when price closes below middle line or ATR-based trailing stop triggers. Works in bull/bear by following strong momentum.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag while capturing major moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for ATR and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # ATR(14) for Keltner bands
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Keltner channels with wider bands (ATR multiplier 2.5)
    ema_center = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_band = ema_center + (2.5 * atr)
    lower_band = ema_center - (2.5 * atr)
    
    # Align Keltner bands to daily timeframe
    upper_band_d = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_d = align_htf_to_ltf(prices, df_1d, lower_band)
    ema_center_d = align_htf_to_ltf(prices, df_1d, ema_center)
    
    # Weekly EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band_d[i]) or np.isnan(lower_band_d[i]) or
            np.isnan(ema_center_d[i]) or np.isnan(ema_50_1w_d[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_band_d[i]
        lower = lower_band_d[i]
        middle = ema_center_d[i]
        weekly_ema = ema_50_1w_d[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above upper band with volume in uptrend (above weekly EMA)
            if price > upper and vol_ok and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume in downtrend (below weekly EMA)
            elif price < lower and vol_ok and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below middle line or weekly EMA turns down
            if price < middle or price < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above middle line or weekly EMA turns up
            if price > middle or price > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WideBand_Keltner_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0