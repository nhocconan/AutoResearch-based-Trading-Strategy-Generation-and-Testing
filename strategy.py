#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR stoploss.
# Long when price breaks above Camarilla R3 level AND 1d volume > 1.4x 20-period average.
# Short when price breaks below Camarilla S3 level AND 1d volume > 1.4x 20-period average.
# Exit on ATR-based stoploss (2.5*ATR from entry) or opposite Camarilla break (R3/S3).
# Uses discrete position size 0.25. Camarilla levels from 1d OHLC provide institutional
# support/resistance that works in both bull and bear markets. Volume confirmation
# reduces false breakouts. ATR stoploss adapts to volatility.
# Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: Camarilla pivots (R3, S3) and Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels from previous 1d bar
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    # Camarilla R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3_1d = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_s3_1d = c_1d - (h_1d - l_1d) * 1.1 / 4
    # Volume spike: current 1d volume > 1.4x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.4 * vol_ma_1d)
    
    # Align 1d indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(atr_4h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # Convert back to boolean
        atr_val = atr_4h_raw[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S3 (opposite breakout)
            if price < s3_level:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R3 (opposite breakout)
            if price > r3_level:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND volume spike
            if price > r3_level and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S3 AND volume spike
            elif price < s3_level and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dVolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0