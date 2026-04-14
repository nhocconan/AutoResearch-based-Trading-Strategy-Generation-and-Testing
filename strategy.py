#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate 12-hour ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12-hour Donchian channels (20-period) - breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1-day 14-period RSI for trend filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(atr[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(rsi_14[i]):
            continue
        
        # Get previous day's data (1d index)
        if i >= 1:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            
            # Calculate Camarilla pivot levels (previous day)
            range_ = prev_high - prev_low
            c = prev_close
            # Camarilla levels
            s3 = c - (range_ * 1.1 / 4)
            r3 = c + (range_ * 1.1 / 4)
            
            # Align Camarilla levels to daily timeframe (constant values for the day)
            s3_array = np.full(len(df_1d), s3)
            r3_array = np.full(len(df_1d), r3)
            s3_1d = align_htf_to_ltf(prices, df_1d, s3_array)[i]
            r3_1d = align_htf_to_ltf(prices, df_1d, r3_array)[i]
            
            # Volume filter: current volume > 1.3x 5-period average
            vol_ma = np.mean(volume[max(0, i-5):i]) if i >= 5 else volume[i]
            
            # Volatility filter: current ATR > 25th percentile of last 50 periods
            vol_filter = True
            if i >= 50:
                vol_percentile = np.percentile(tr[max(0, i-50):i+1], 25)
                vol_filter = atr[i] > vol_percentile
            
            # Trend filter: daily RSI between 40 and 60 (avoid extremes)
            rsi_filter = (rsi_14[i] >= 40) & (rsi_14[i] <= 60)
            
            if position == 0:
                # Long: Price breaks above R3 with volume, volatility, and RSI filter
                if (close[i] > r3_1d and close[i-1] <= r3_1d and 
                    volume[i] > vol_ma * 1.3 and 
                    close[i] > donchian_high[i] and  # Breakout confirmation
                    vol_filter and rsi_filter):
                    position = 1
                    signals[i] = position_size
                # Short: Price breaks below S3 with volume, volatility, and RSI filter
                elif (close[i] < s3_1d and close[i-1] >= s3_1d and 
                      volume[i] > vol_ma * 1.3 and 
                      close[i] < donchian_low[i] and  # Breakdown confirmation
                      vol_filter and rsi_filter):
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price breaks below S3 (reverse signal) or drops below Donchian low
                if close[i] < s3_1d or close[i] < donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price breaks above R3 (reverse signal) or rises above Donchian high
                if close[i] > r3_1d or close[i] > donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3_Donchian_Volume_RSI_Filter"
timeframe = "12h"
leverage = 1.0