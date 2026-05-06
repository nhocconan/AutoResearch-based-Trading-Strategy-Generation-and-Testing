#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and ATR volatility filter
# Breakout above 12h Donchian high or below low with volume > 2x 20-period average indicates strong momentum
# ATR filter ensures trades occur during sufficient volatility (ATR > 1.5x 50-period ATR mean)
# Works in bull/bear markets: breakouts capture trends, volatility filter avoids choppy periods
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "4h_DonchianBreakout_12h_VolumeATRFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 20-period Donchian high and low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: >2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # ATR volatility filter: ATR > 1.5x 50-period ATR mean
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_mean_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr > (1.5 * atr_mean_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(atr_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 12h Donchian high with volume and volatility
            if close[i] > donchian_high_aligned[i] and volume_filter[i] and atr_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 12h Donchian low with volume and volatility
            elif close[i] < donchian_low_aligned[i] and volume_filter[i] and atr_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 12h Donchian low (failed breakout) or ATR drops
            if close[i] < donchian_low_aligned[i] or not atr_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 12h Donchian high (failed breakdown) or ATR drops
            if close[i] > donchian_high_aligned[i] or not atr_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals