#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Reversal with Volume Confirmation
# Camarilla levels from prior 1d: R3/H4, S3/L4 for reversals, R4/H5 and S4/L5 for breakouts
# Fade at R3/S3 when price shows rejection (close near open) + volume spike
# Breakout continuation at R4/S4 when price closes beyond level with volume surge
# Works in ranging markets (fade) and trending markets (breakout)
# Uses discrete sizing (0.25) to limit overtrading and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get prior 1-day OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    # Need complete prior day, so we use shift(1) on the daily data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for prior day
    # R4 = Close + 1.5*(High-Low), R3 = Close + 1.1*(High-Low)
    # S3 = Close - 1.1*(High-Low), S4 = Close - 1.5*(High-Low)
    rang = high_1d - low_1d
    r4 = close_1d + 1.5 * rang
    r3 = close_1d + 1.1 * rang
    s3 = close_1d - 1.1 * rang
    s4 = close_1d - 1.5 * rang
    
    # Align levels to 6h timeframe (these levels are valid until next 1d bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current > 2.0x median of last 24 bars (4 days)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    # Price action signals
    body_size = np.abs(close - open_)
    candle_range = high - low
    # Avoid division by zero
    body_ratio = np.where(candle_range > 0, body_size / candle_range, 0)
    # Rejection signal: small body relative to range (pin bar, doji)
    is_rejection = body_ratio < 0.3
    
    signals = np.zeros(n)
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Fade at R3/S3: rejection + volume spike
        if (close[i] >= r3_aligned[i] and close[i] <= r4_aligned[i] and
            is_rejection[i] and volume[i] > vol_threshold[i]):
            # Price rejected at R3 level -> short
            signals[i] = -0.25
        elif (close[i] <= s3_aligned[i] and close[i] >= s4_aligned[i] and
              is_rejection[i] and volume[i] > vol_threshold[i]):
            # Price rejected at S3 level -> long
            signals[i] = 0.25
        
        # Breakout continuation at R4/S4: close beyond level + volume spike
        elif (close[i] > r4_aligned[i] and volume[i] > vol_threshold[i]):
            # Broke above R4 with strength -> long
            signals[i] = 0.25
        elif (close[i] < s4_aligned[i] and volume[i] > vol_threshold[i]):
            # Broke below S4 with strength -> short
            signals[i] = -0.25
        
        # Exit conditions: return to middle zone or lose momentum
        elif i > 0 and signals[i-1] != 0:
            prev_signal = signals[i-1]
            # Exit long if price returns below R3 or loses volume/momentum
            if prev_signal == 0.25:
                if (close[i] < r3_aligned[i] or 
                    volume[i] <= vol_threshold[i] * 0.5):
                    signals[i] = 0.0
                else:
                    signals[i] = prev_signal
            # Exit short if price returns above S3 or loses volume/momentum
            else:  # prev_signal == -0.25
                if (close[i] > s3_aligned[i] or 
                    volume[i] <= vol_threshold[i] * 0.5):
                    signals[i] = 0.0
                else:
                    signals[i] = prev_signal
        
        # Otherwise hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Camarilla_Pivot_Reversal_Breakout"
timeframe = "6h"
leverage = 1.0