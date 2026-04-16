#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R1 AND 12h EMA50 trending up AND volume > 2.0x 20-period average.
# Short when price breaks below Camarilla S1 AND 12h EMA50 trending down AND volume > 2.0x 20-period average.
# Exit on opposite Camarilla break (S1 for longs, R1 for shorts).
# Uses discrete position size 0.25. Designed to capture institutional breakouts with volume confirmation.
# Camarilla levels from 1d provide strong support/resistance; EMA50 on 12h filters counter-trend noise.
# Volume spike >2.0x average ensures institutional participation, reducing false breakouts.
# Target: 100-180 total trades over 4 years (25-45/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla R1, S1 from 1d OHLC ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_upper = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_lower = close_1d - 1.1 * (high_1d - low_1d) / 12
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_1d, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_1d, camarilla_lower)
    
    # === 12h Indicators: EMA50 for trend ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_up = ema_50_12h_aligned > np.roll(ema_50_12h_aligned, 1)
    ema_down = ema_50_12h_aligned < np.roll(ema_50_12h_aligned, 1)
    
    # === 12h Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (2.0 * vol_ma_12h_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 60 periods needed for EMA50/volume MA)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_lower_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_upper_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND EMA50 trending up AND volume spike
            if price > camarilla_upper_aligned[i] and ema_up[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND EMA50 trending down AND volume spike
            elif price < camarilla_lower_aligned[i] and ema_down[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1_S1_12hEMA50_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0