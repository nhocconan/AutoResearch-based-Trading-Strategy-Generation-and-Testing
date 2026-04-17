#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA200 trend filter.
- Long when price closes above 4h Donchian(20) high + volume > 1.5x 20-period 1h volume MA + price above 1d EMA200
- Short when price closes below 4h Donchian(20) low + volume > 1.5x 20-period 1h volume MA + price below 1d EMA200
- Fixed position size 0.20 to limit fee churn and manage drawdown
- No trailing stop; exit on opposite Donchian breakout or trend reversal
- Designed for low trade frequency (target: 60-150 total trades over 4 years) to avoid fee drag
- Works in bull markets (buying above 4h Donchian high with 1d EMA200 uptrend) and bear markets (selling below 4h Donchian low with 1d EMA200 downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (HTF)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian(20)
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # Get 1d data for EMA200 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 1h data for volume confirmation (primary timeframe)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for Donchian(20) and EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        dh = donch_high_aligned[i]
        dl = donch_low_aligned[i]
        ema_200 = ema_200_aligned[i]
        vol_ma = volume_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 1d EMA200 trend filter
            # Long: price closes above 4h Donchian high + volume spike + price above 1d EMA200
            if price > dh and vol > 1.5 * vol_ma and price > ema_200:
                signals[i] = 0.20
                position = 1
            # Short: price closes below 4h Donchian low + volume spike + price below 1d EMA200
            elif price < dl and vol > 1.5 * vol_ma and price < ema_200:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long on Donchian low break or trend reversal
            if price < dl or price < ema_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short on Donchian high break or trend reversal
            if price > dh or price > ema_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hBreakout_1dEMA200_VolumeSpike"
timeframe = "1h"
leverage = 1.0