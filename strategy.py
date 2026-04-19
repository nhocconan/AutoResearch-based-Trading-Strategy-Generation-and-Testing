#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and volume confirmation.
# Uses daily ATR to filter out low volatility periods where breakouts fail.
# In high volatility (ATR > 1.5x 20-day mean ATR): long on Donchian(20) breakout above, short on breakdown below.
# Volume confirmation: volume > 1.5x 20-period average.
# Target: 20-40 trades/year per symbol to stay within frequency limits.
# Works in both bull (catch breakouts) and bear (catch breakdowns) markets.
name = "4h_Donchian20_1dATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Calculate ATR (14-period Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilder_smooth(tr, 14)
    # Calculate 20-period average ATR for volatility filter
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for Donchian channels (using daily high/low)
    high_1d_dc = df_1d['high'].values
    low_1d_dc = df_1d['low'].values
    
    # Donchian channels (20-period) on daily data
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donch_high_dc = rolling_max(high_1d_dc, 20)
    donch_low_dc = rolling_min(low_1d_dc, 20)
    
    # Get 4h data for entry Donchian channels (breakout levels)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels (20-period) on 4h data for entry signals
    donch_high_4h = rolling_max(high_4h, 20)
    donch_low_4h = rolling_min(low_4h, 20)
    
    # Align indicators to 4h timeframe
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    donch_high_dc_aligned = align_htf_to_ltf(prices, df_1d, donch_high_dc)
    donch_low_dc_aligned = align_htf_to_ltf(prices, df_1d, donch_low_dc)
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # Get 4h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure ATR (14+20), Donchian (20), and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_ma_20_aligned[i]) or np.isnan(donch_high_4h_aligned[i]) or 
            np.isnan(donch_low_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_ma = atr_ma_20_aligned[i]
        atr_val = atr_14_aligned[i]
        donch_high_4h = donch_high_4h_aligned[i]
        donch_low_4h = donch_low_4h_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volatility filter: only trade when ATR > 1.5x 20-day average ATR
        volatility_filter = atr_val > 1.5 * atr_ma
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long on breakout above 4h Donchian high with volatility and volume confirmation
            if volatility_filter and volume_confirmed and price > donch_high_4h:
                signals[i] = 0.25
                position = 1
            # Enter short on breakdown below 4h Donchian low with volatility and volume confirmation
            elif volatility_filter and volume_confirmed and price < donch_low_4h:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below 4h Donchian low
            if price < donch_low_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above 4h Donchian high
            if price > donch_high_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals