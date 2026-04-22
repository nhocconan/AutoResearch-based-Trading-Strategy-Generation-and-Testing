#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA trend filter and volume confirmation
# Strategy uses Camarilla pivot levels from daily data to identify key support/resistance levels.
# Breakouts above R1 or below S1 trigger entries, filtered by 12h EMA trend direction.
# Volume confirmation (current volume > 1.5x 20-period average) filters false breakouts.
# Designed for 4h timeframe targeting 20-40 trades/year. Works in bull markets by capturing
# upward breakouts and in bear markets by avoiding false breakdowns via trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot levels (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 12h data for trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period volume average for spike detection
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using previous day's OHLC
        if i < 1:  # Need at least one day of data
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get previous day's OHLC (assuming daily data aligned to 4h)
        prev_day_idx = i - 6  # Approximately 6*4h = 1 day back
        if prev_day_idx < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # For simplicity, we'll use the 1d data directly for Camarilla calculation
        # In practice, we need to get the actual daily OHLC values
        # Since we don't have direct access to previous day's OHLC in the loop,
        # we'll use a simplified approach: calculate Camarilla from 1d data and align
        
        # Calculate Camarilla levels from 1d data
        # We need to access the 1d OHLC data properly
        # Let's get the 1d data values
        if len(df_1d) > 0:
            # Get the most recent completed daily bar
            # For 4h data, we need to map to daily bars
            day_index = i // 6  # 6 four-hour bars per day
            if day_index >= len(df_1d) or day_index < 1:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
                    
            # Get previous day's OHLC (completed day)
            prev_day = day_index - 1
            if prev_day < 0:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
                
            # Get OHLC values for previous day
            try:
                # Access values from df_1d - assuming it has the standard columns
                high_prev = df_1d['high'].iloc[prev_day]
                low_prev = df_1d['low'].iloc[prev_day]
                close_prev = df_1d['close'].iloc[prev_day]
            except:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
            
            # Calculate Camarilla levels
            range_prev = high_prev - low_prev
            if range_prev <= 0:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
                
            # Camarilla levels
            R1 = close_prev + (range_prev * 1.1 / 12)
            S1 = close_prev - (range_prev * 1.1 / 12)
            R2 = close_prev + (range_prev * 1.1 / 6)
            S2 = close_prev - (range_prev * 1.1 / 6)
            R3 = close_prev + (range_prev * 1.1 / 4)
            S3 = close_prev - (range_prev * 1.1 / 4)
            R4 = close_prev + (range_prev * 1.1 / 2)
            S4 = close_prev - (range_prev * 1.1 / 2)
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 12h uptrend + volume spike
            if (close[i] > R1 and 
                close[i] > ema_34_12h_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 12h downtrend + volume spike
            elif (close[i] < S1 and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite S1/R1 or trend reversal
            if position == 1:
                # Exit on return to S1 or trend reversal
                if (close[i] <= S1 or 
                    close[i] < ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on return to R1 or trend reversal
                if (close[i] >= R1 or 
                    close[i] > ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0