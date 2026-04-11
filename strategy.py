#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elliott_wave_oscillator_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Elliott Wave Oscillator (EWO)
    # EWO = SMA(5) - SMA(34) on daily closes
    sma5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    sma34 = pd.Series(close_1d).rolling(window=34, min_periods=34).mean().values
    ewo = sma5 - sma34
    
    # Shift to avoid look-ahead (use previous day's EWO for current day)
    ewo = np.roll(ewo, 1)
    ewo[0] = np.nan
    
    # Align EWO to 6h timeframe
    ewo_aligned = align_htf_to_ltf(prices, df_1d, ewo)
    
    # Calculate daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6-period average volume for confirmation (on daily)
    vol_avg_6 = pd.Series(volume_1d).rolling(window=6, min_periods=6).mean().values
    vol_avg_6_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_6)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 40 to ensure sufficient data
    for i in range(40, n):
        # Skip if any required data is invalid
        if (np.isnan(ewo_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_avg_6_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current daily volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_6_aligned[i]
        
        price = close[i]
        
        # EWO conditions with volume confirmation
        # Long when EWO crosses above 0 with volume (bullish wave)
        long_signal = (ewo_aligned[i] > 0) and vol_confirm
        # Short when EWO crosses below 0 with volume (bearish wave)
        short_signal = (ewo_aligned[i] < 0) and vol_confirm
        
        # Volatility filter: avoid trading in extremely low volatility
        # Only trade when ATR is above 30% of its 20-period average
        atr_ma_20 = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean()
        atr_ma_20_val = atr_ma_20.iloc[i] if hasattr(atr_ma_20, 'iloc') else atr_ma_20[i] if i < len(atr_ma_20) else np.nan
        vol_filter = not np.isnan(atr_ma_20_val) and atr_1d_aligned[i] > (0.3 * atr_ma_20_val)
        
        if long_signal and vol_filter and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and vol_filter and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (ewo_aligned[i] < 0 or not vol_filter):
            # Exit long when EWO turns bearish or volatility drops
            position = 0
            signals[i] = 0.0
        elif position == -1 and (ewo_aligned[i] > 0 or not vol_filter):
            # Exit short when EWO turns bullish or volatility drops
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals