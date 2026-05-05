#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter
# Long when price breaks above Donchian upper band (20-period high) AND volume > 1.5x 20-period average AND 1d EMA50 trending up
# Short when price breaks below Donchian lower band (20-period low) AND volume > 1.5x 20-period average AND 1d EMA50 trending down
# Exit when price crosses back to the midpoint of the Donchian channel OR 1d EMA50 flips direction
# Uses discrete sizing (0.30) to balance profit potential and risk management.
# Donchian channels provide clear trend-following structure, volume confirms conviction,
# 1d EMA50 filters for primary trend to avoid counter-trend whipsaws in choppy markets.
# Designed to work in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_VolumeSpike_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels (using lookback period to avoid look-ahead)
    # Upper band: highest high of last 20 periods (excluding current)
    # Lower band: lowest low of last 20 periods (excluding current)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_prev = np.concatenate([[np.nan], ema_50[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA50 > previous EMA50
    uptrend_1d = ema_50 > ema_50_prev
    downtrend_1d = ema_50 < ema_50_prev
    
    # Align 1d trend to 4h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND volume spike AND 1d uptrend
            if (close[i] > donchian_high[i] and 
                volume_filter[i] and 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian lower band AND volume spike AND 1d downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  downtrend_1d_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian midpoint OR 1d trend flips to downtrend
            if (close[i] < donchian_mid[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back to Donchian midpoint OR 1d trend flips to uptrend
            if (close[i] > donchian_mid[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals