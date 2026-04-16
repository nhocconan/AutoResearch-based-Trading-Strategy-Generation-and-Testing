#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND price > 4h EMA34 (uptrend) AND volume > 1.8x 24-period average.
# Short when price breaks below Camarilla S3 AND price < 4h EMA34 (downtrend) AND volume > 1.8x 24-period average.
# Uses discrete position size 0.20. Camarilla levels provide intraday support/resistance, 4h EMA34 ensures alignment with higher timeframe trend,
# volume spike confirms participation. Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 60-150 trades over 4 years (15-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Calculate daily pivot from previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align daily Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1h Indicators: Volume Spike (volume > 1.8x 24-period average) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Get 4h data once before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need enough for EMA34 calculation
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: EMA34 for trend filter ===
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA34 to 1h timeframe
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA, 24 for volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema_4h = ema_34_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Camarilla S3 or volume spike ends
            if price < s3 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Camarilla R3 or volume spike ends
            if price > r3 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND price > 4h EMA34 (uptrend) AND volume spike
            if price > r3 and price > ema_4h and vol_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT: Price breaks below Camarilla S3 AND price < 4h EMA34 (downtrend) AND volume spike
            elif price < s3 and price < ema_4h and vol_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_Camarilla_R3_S3_4hEMA34_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0