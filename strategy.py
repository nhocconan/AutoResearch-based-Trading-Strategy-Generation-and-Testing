# SPDX-License-Identifier: MIT
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
    
    # Calculate 6-hour Donchian channels (20-period) - breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1-day ADX for trend strength (14-period)
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    high_close_1d[0] = high_low_1d[0]
    low_close_1d[0] = high_low_1d[0]
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    
    # Directional movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-day 20-period EMA for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(atr[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(adx[i]) or np.isnan(ema_20_1d[i]):
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
            s4 = prev_low - 3 * (prev_high - prev_low)
            r4 = prev_high + 3 * (prev_high - prev_low)
            
            # Align pivot levels to daily timeframe (constant values for the day)
            s3_array = np.full(len(df_1d), s3)
            r3_array = np.full(len(df_1d), r3)
            s4_array = np.full(len(df_1d), s4)
            r4_array = np.full(len(df_1d), r4)
            s3_1d = align_htf_to_ltf(prices, df_1d, s3_array)[i]
            r3_1d = align_htf_to_ltf(prices, df_1d, r3_array)[i]
            s4_1d = align_htf_to_ltf(prices, df_1d, s4_array)[i]
            r4_1d = align_htf_to_ltf(prices, df_1d, r4_array)[i]
            
            # Volume filter: current volume > 1.5x 5-period average
            vol_ma = np.mean(volume[max(0, i-5):i]) if i >= 5 else volume[i]
            
            # Volatility filter: current ATR > 30th percentile of last 50 periods
            vol_filter = True
            if i >= 50:
                vol_percentile = np.percentile(tr[max(0, i-50):i+1], 30)
                vol_filter = atr[i] > vol_percentile
            
            # Trend filter: daily ADX > 25 AND price above/below daily EMA20
            trend_filter = adx[i] > 25
            
            if position == 0:
                # Long: Price breaks above R3 with volume, volatility, and trend filter
                # Additional filter: price above daily EMA20 for long bias
                if (close[i] > r3_1d and close[i-1] <= r3_1d and 
                    volume[i] > vol_ma * 1.5 and 
                    close[i] > donchian_high[i] and  # Breakout confirmation
                    vol_filter and trend_filter and 
                    close[i] > ema_20_1d[i]):  # Above daily EMA20
                    position = 1
                    signals[i] = position_size
                # Short: Price breaks below S3 with volume, volatility, and trend filter
                # Additional filter: price below daily EMA20 for short bias
                elif (close[i] < s3_1d and close[i-1] >= s3_1d and 
                      volume[i] > vol_ma * 1.5 and 
                      close[i] < donchian_low[i] and  # Breakdown confirmation
                      vol_filter and trend_filter and 
                      close[i] < ema_20_1d[i]):  # Below daily EMA20
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price breaks below S4 (strong reversal) or drops below daily EMA20
                if close[i] < s4_1d or close[i] < ema_20_1d[i]:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price breaks above R4 (strong reversal) or rises above daily EMA20
                if close[i] > r4_1d or close[i] > ema_20_1d[i]:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "6h_Pivot_S3R3_ADX_EMA20_Filter"
timeframe = "6h"
leverage = 1.0