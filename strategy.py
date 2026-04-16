#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR stoploss.
# Long when price breaks above Camarilla R3 level AND 1d volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S3 level AND 1d volume > 1.5x 20-period average.
# Exit on ATR-based stoploss (2.5*ATR from entry) or opposite Camarilla break (R3/S3).
# Uses discrete position size 0.25. Camarilla levels derived from prior 1d OHLC.
# Volume confirmation filters breakouts during low-activity periods.
# ATR stoploss adapts to volatility. Designed for fewer trades (~12-25/year) to minimize fee drag.
# Works in both bull and bear markets by requiring volume and using symmetric pivot structure.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: Camarilla levels (R3, S3) and Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    # Camarilla: based on prior day's OHLC
    cam_r3 = ((df_1d['high'] - df_1d['low']) * 1.1 / 4) + df_1d['close']
    cam_s3 = df_1d['close'] - ((df_1d['high'] - df_1d['low']) * 1.1 / 4)
    # Align to 12h timeframe (use prior day's levels)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3.values)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3.values)
    # Volume confirmation: current 1d volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
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
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_12h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_12h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S3 (opposite breakout)
            if price < cam_s3_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R3 (opposite breakout)
            if price > cam_r3_aligned[i]:
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
            if price > cam_r3_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S3 AND volume spike
            elif price < cam_s3_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R3_S3_1dVolumeSpike_ATRStop_V1"
timeframe = "12h"
leverage = 1.0