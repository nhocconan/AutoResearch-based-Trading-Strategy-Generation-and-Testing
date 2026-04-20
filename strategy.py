#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian breakout with volume confirmation and volatility filter.
# Long when price breaks above daily Donchian high (20-day) with volume > 1.5x average.
# Short when price breaks below daily Donchian low (20-day) with volume > 1.5x average.
# Exit when price returns to the opposite Donchian band.
# Volume and volatility filters reduce false breakouts. Designed for 12h timeframe to capture multi-day trends
# while avoiding overtrading. Works in both bull and bear markets by trading breakouts in direction of trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for HTF regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Daily ATR(14) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 12h price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in HTF indicators
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        vol_ma_val = volume_ma_20_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume must be above 1.5x 20-day average
        vol_filter = vol > 1.5 * vol_ma_val
        
        # Volatility filter: only trade when volatility is below 80th percentile (avoid extreme volatility)
        vol_regime = atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 80)
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume confirmation
            if price > upper_band and vol_filter and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume confirmation
            elif price < lower_band and vol_filter and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to or below lower Donchian band
            if price <= lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to or above upper Donchian band
            if price >= upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeVolatilityFilter_Session"
timeframe = "12h"
leverage = 1.0