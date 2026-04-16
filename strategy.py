#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA200 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND price > 1w EMA200 (uptrend) AND volume > 1.8x 24-period average.
# Short when price breaks below Camarilla S3 AND price < 1w EMA200 (downtrend) AND volume > 1.8x 24-period average.
# Uses discrete position size 0.25. Camarilla provides institutional pivot levels, 1w EMA200 ensures alignment with higher timeframe trend,
# volume spike confirms institutional participation. Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 50-150 trades over 4 years (12-37/year) to balance opportunity and fee drag for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (based on previous bar's OHLC) ===
    # Calculate using previous bar's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current values (will be filtered by warmup)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4.0)  # R3
    s3 = pivot - (range_hl * 1.1 / 4.0)  # S3
    
    # === 12h Indicators: Volume Spike (volume > 1.8x 24-period average) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:  # Need enough for EMA200 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA200 for trend filter ===
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 12h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 200 periods needed for EMA, 24 for volume MA)
    warmup = 220
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r3_level = r3[i]
        s3_level = s3[i]
        ema_1w = ema_200_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Camarilla S3 or volume spike ends
            if price < s3_level or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Camarilla R3 or volume spike ends
            if price > r3_level or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND price > 1w EMA200 (uptrend) AND volume spike
            if price > r3_level and price > ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S3 AND price < 1w EMA200 (downtrend) AND volume spike
            elif price < s3_level and price < ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_CamarillaR3S3_1wEMA200_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0