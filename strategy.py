#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h chart with 1d Parabolic SAR trend filter and 12h Donchian breakout.
# Long when price breaks above Donchian(20) upper band and SAR indicates uptrend (SAR < close).
# Short when price breaks below Donchian(20) lower band and SAR indicates downtrend (SAR > close).
# Uses daily SAR to filter trend direction and avoid counter-trend trades.
# Target: 15-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Parabolic SAR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Parabolic SAR calculation (standard parameters: af=0.02, max_af=0.2)
    # Initialize
    sar = np.zeros_like(high_1d)
    trend = np.ones_like(high_1d)  # 1 for uptrend, -1 for downtrend
    af = 0.02
    max_af = 0.2
    ep = high_1d[0]  # extreme point
    sar[0] = low_1d[0]
    
    for i in range(1, len(high_1d)):
        if trend[i-1] == 1:  # was uptrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if sar[i] > low_1d[i]:  # trend reversal
                trend[i] = -1
                sar[i] = ep
                ep = low_1d[i]
                af = 0.02
            else:
                trend[i] = 1
                if high_1d[i] > ep:
                    ep = high_1d[i]
                    af = min(af + 0.02, max_af)
        else:  # was downtrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if sar[i] < high_1d[i]:  # trend reversal
                trend[i] = 1
                sar[i] = ep
                ep = high_1d[i]
                af = 0.02
            else:
                trend[i] = -1
                if low_1d[i] < ep:
                    ep = low_1d[i]
                    af = min(af + 0.02, max_af)
    
    # Align SAR to 12h timeframe (need trend direction, not SAR value)
    # Uptrend = 1, Downtrend = -1
    sar_trend = trend
    sar_trend_aligned = align_htf_to_ltf(prices, df_1d, sar_trend)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(sar_trend_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        trend_up = sar_trend_aligned[i] == 1
        trend_down = sar_trend_aligned[i] == -1
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above upper band, uptrend, volume
            if price > upper_band and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band, downtrend, volume
            elif price < lower_band and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower band or trend turns down
            if price < lower_band or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper band or trend turns up
            if price > upper_band or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_SAR_Trend_Donchian20_Breakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0