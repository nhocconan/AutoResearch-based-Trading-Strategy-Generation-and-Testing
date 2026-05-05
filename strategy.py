#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R4 level AND price > EMA34(1d) AND volume > 2.5x 20-period average
# Short when price breaks below Camarilla S4 level AND price < EMA34(1d) AND volume > 2.5x 20-period average
# Exit when price crosses back below Camarilla S4 (for longs) or above R4 (for shorts) OR trend flips (price crosses EMA34(1d))
# Camarilla R4/S4 levels provide stronger support/resistance than R3/S3, reducing false breakouts
# 1d EMA34 provides medium timeframe trend filter to avoid counter-trend whipsaws
# Higher volume threshold (2.5x) ensures only significant institutional participation triggers entries
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 12h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "12h_Camarilla_R4_S4_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels: based on previous day's OHLC
    # R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    camarilla_r4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_s4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 2.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 AND price > EMA34(1d) AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S4 AND price < EMA34(1d) AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Camarilla S4 (mean reversion) OR price < EMA34(1d) (trend flip)
            if (close[i] < camarilla_s4_aligned[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Camarilla R4 (mean reversion) OR price > EMA34(1d) (trend flip)
            if (close[i] > camarilla_r4_aligned[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals