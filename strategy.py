# 6H_Pivot_S3R3_Breakout_VolATR_Filter
# Hypothesis: Price breaking through key daily support/resistance (S3/R3) with volume and volatility confirmation captures institutional breakout moves in both bull and bear markets. The daily pivot levels provide meaningful reference points, while volume and ATR filters ensure trades occur during genuine momentum shifts rather than noise.
# Timeframe: 6h balances noise reduction with sufficient trade frequency (target: 12-37 trades/year).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6-hour Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(atr[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        
        # Get previous day's data (1d index)
        if i >= 1:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            
            # Calculate pivot points (standard formula)
            pivot = (prev_high + prev_low + prev_close) / 3.0
            s1 = (2 * pivot) - prev_high
            r1 = (2 * pivot) - prev_low
            s2 = pivot - (prev_high - prev_low)
            r2 = pivot + (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            r3 = prev_high + 2 * (pivot - prev_low)
            
            # Align S3/R3 to daily timeframe (constant values for the day)
            s3_array = np.full(len(df_1d), s3)
            r3_array = np.full(len(df_1d), r3)
            s3_1d = align_htf_to_ltf(prices, df_1d, s3_array)[i]
            r3_1d = align_htf_to_ltf(prices, df_1d, r3_array)[i]
            
            # Volume filter: current volume > 1.5x 5-period average
            vol_ma = np.mean(volume[max(0, i-5):i]) if i >= 5 else volume[i]
            
            # Volatility filter: current ATR > 30th percentile of last 50 periods
            vol_filter = True
            if i >= 50:
                vol_percentile = np.percentile(tr[max(0, i-50):i+1], 30)
                vol_filter = atr[i] > vol_percentile
            
            if position == 0:
                # Long: Price breaks above R3 with volume and volatility filter
                if (close[i] > r3_1d and close[i-1] <= r3_1d and 
                    volume[i] > vol_ma * 1.5 and 
                    close[i] > donchian_high[i] and  # Additional breakout confirmation
                    vol_filter):
                    position = 1
                    signals[i] = position_size
                # Short: Price breaks below S3 with volume and volatility filter
                elif (close[i] < s3_1d and close[i-1] >= s3_1d and 
                      volume[i] > vol_ma * 1.5 and 
                      close[i] < donchian_low[i] and  # Additional breakdown confirmation
                      vol_filter):
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price breaks below S3 or volatility drops significantly
                if close[i] < s3_1d:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price breaks above R3
                if close[i] > r3_1d:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "6h_Pivot_S3R3_Breakout_VolATR_Filter"
timeframe = "6h"
leverage = 1.0