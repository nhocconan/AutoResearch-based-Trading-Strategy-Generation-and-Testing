#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Long when price > Alligator Jaw (13-period SMMA) AND Alligator Mouth is open (Jaw > Teeth > Lips) AND 1w uptrend AND volume spike (>1.8x 50-period volume MA).
# Short when price < Alligator Lips AND Alligator Mouth is open (Lips < Teeth < Jaw) AND 1w downtrend AND volume spike.
# Uses 1w EMA50 for higher timeframe trend alignment to avoid counter-trend trades.
# Volume spike confirms institutional participation. Designed for 12h timeframe to achieve 50-150 total trades over 4 years.
# Williams Alligator identifies trend absence/presence and direction via smoothed moving averages.

name = "12h_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on primary timeframe (12h)
    # Smoothed Moving Average (SMMA) = EMA with alpha = 1/period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=np.float64)
        result = np.empty_like(data)
        result[:] = np.nan
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator components: Jaw (13), Teeth (8), Lips (5)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Get 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection (50-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (volume_ma * 1.8)  # Volume at least 1.8x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        open_val = open_prices[i]
        vol_spike = volume_spike[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        trend_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        
        # Alligator Mouth open conditions
        mouth_open_up = jaw_val > teeth_val > lips_val   # Bullish alignment
        mouth_open_down = lips_val < teeth_val < jaw_val  # Bearish alignment
        
        if position == 0:
            # Long: price > Jaw AND mouth open bullish AND 1w uptrend AND volume spike
            if close_val > jaw_val and mouth_open_up and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < Lips AND mouth open bearish AND 1w downtrend AND volume spike
            elif close_val < lips_val and mouth_open_down and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: price closes below Teeth (trend weakening)
            if close_val < teeth_val:
                exit_signal = True
            # Exit: 1w trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            # Exit: Alligator mouth closes (loss of momentum)
            elif not mouth_open_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: price closes above Teeth (trend weakening)
            if close_val > teeth_val:
                exit_signal = True
            # Exit: 1w trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            # Exit: Alligator mouth closes (loss of momentum)
            elif not mouth_open_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals