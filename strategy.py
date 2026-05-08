#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation.
# Long when price breaks above upper band AND price > EMA34(1d) AND volume > 1.8x 20-period average.
# Short when price breaks below lower band AND price < EMA34(1d) AND volume > 1.8x 20-period average.
# Exit when price crosses back below upper band (long) or above lower band (short).
# Bollinger squeeze (low volatility) precedes explosive moves. EMA34(1d) filters trend direction.
# Volume confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_BollingerSqueeze_1dEMA34_Volume"
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
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma20)
    
    # Bollinger Bands (20, 2) on 6h close
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper_band = sma + (std * bb_std)
    lower_band = sma - (std * bb_std)
    upper_band = upper_band.values
    lower_band = lower_band.values
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # EMA34 on 1d close
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA34 and Bollinger Bands
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper band, price > EMA34, volume filter
            long_cond = (close[i] > upper_band[i]) and (close[i] > ema_34_aligned[i]) and volume_filter[i]
            # Short conditions: break below lower band, price < EMA34, volume filter
            short_cond = (close[i] < lower_band[i]) and (close[i] < ema_34_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross back below upper band
            if close[i] < upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross back above lower band
            if close[i] > lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals