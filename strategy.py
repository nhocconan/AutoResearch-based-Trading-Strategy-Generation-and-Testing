#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h trend filter and volume confirmation
# Long when price breaks above Camarilla R4 AND 12h EMA(21) is rising AND volume > 1.5x average
# Short when price breaks below Camarilla S4 AND 12h EMA(21) is falling AND volume > 1.5x average
# Exit when price crosses back through Camarilla pivot point (mean reversion) or opposite breakout
# Camarilla levels from 1d provide intraday support/resistance; 12h EMA ensures intermediate trend alignment; volume confirms institutional interest
# Designed to work in both bull and bear markets by following the dominant trend on 12h timeframe
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels from 1d (using previous day's high, low, close)
    # We need to shift by 1 to avoid look-ahead: use previous day's data for today's levels
    prev_high = pd.Series(high).shift(1)
    prev_low = pd.Series(low).shift(1)
    prev_close = pd.Series(close).shift(1)
    
    # Camarilla levels calculation
    R4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    R2 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    R1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    PP = (prev_high + prev_low + prev_close) / 3
    S1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    S2 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    S4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Calculate EMA on 12h (21-period) for trend filter
    ema_21 = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(R4.iloc[i]) or 
            np.isnan(S4.iloc[i]) or 
            np.isnan(PP.iloc[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get EMA values aligned to 6h timeframe
        ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21.values)
        ema_val = ema_21_aligned[i]
        ema_prev = ema_21_aligned[i-1]
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above Camarilla R4 AND 12h EMA rising AND volume confirmation
            if (high_val > R4.iloc[i] and ema_val > ema_prev and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Camarilla S4 AND 12h EMA falling AND volume confirmation
            elif (low_val < S4.iloc[i] and ema_val < ema_prev and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot point OR opposite breakout
            if (close_val < PP.iloc[i] or 
                low_val < S4.iloc[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot point OR opposite breakout
            if (close_val > PP.iloc[i] or 
                high_val > R4.iloc[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Camarilla_12hEMA_Volume"
timeframe = "6h"
leverage = 1.0