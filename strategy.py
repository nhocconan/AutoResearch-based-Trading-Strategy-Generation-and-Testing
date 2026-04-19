#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily trend filter and volume confirmation.
# Long when price breaks above 12h upper band with price above daily EMA50 and volume spike (>1.5x average).
# Short when price breaks below 12h lower band with price below daily EMA50 and volume spike.
# Uses daily EMA50 as trend filter to avoid counter-trend trades and reduce whipsaw in sideways markets.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 12-37 trades/year per symbol (~50-150 total over 4 years).
name = "12h_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (wait for daily close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 12h data
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 50, 20)  # Need Donchian, EMA, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = upper[i]
        lower_band = lower[i]
        ema_trend = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above upper band AND above daily EMA50
            if price > upper_band and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band AND below daily EMA50
            elif price < lower_band and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below lower band or below daily EMA50
            if price < lower_band or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above upper band or above daily EMA50
            if price > upper_band or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals