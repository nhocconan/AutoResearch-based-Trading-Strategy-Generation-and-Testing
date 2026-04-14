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
    
    # Load weekly and daily data once before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly ATR for volatility filter (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.roll(close_1w, 1))
    low_close = np.abs(low_1w - np.roll(close_1w, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_series = pd.Series(tr)
    atr_1w = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6-hour ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly Donchian channels (20-period) - breakout levels
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    donchian_high_1w = high_series_1w.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_1w = low_series_1w.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate weekly moving average for trend filter
    ma_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Calculate daily RSI for trend filter
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
        if np.isnan(atr[i]) or np.isnan(atr_1w[i]) or np.isnan(donchian_high_1w[i]) or np.isnan(donchian_low_1w[i]) or np.isnan(ma_1w[i]) or np.isnan(rsi_14[i]):
            continue
        
        # Get previous week's data (1w index)
        if i >= 1:
            prev_high_1w = high_1w[i-1]
            prev_low_1w = low_1w[i-1]
            prev_close_1w = close_1w[i-1]
            
            # Calculate weekly ATR-based levels (similar to ATR channels)
            atr_value = atr_1w[i-1]
            upper_channel = prev_close_1w + 1.5 * atr_value
            lower_channel = prev_close_1w - 1.5 * atr_value
            
            # Align weekly levels to 6h timeframe
            upper_channel_array = np.full(len(df_1w), upper_channel)
            lower_channel_array = np.full(len(df_1w), lower_channel)
            ma_1w_array = ma_1w
            
            upper_channel_6h = align_htf_to_ltf(prices, df_1w, upper_channel_array)[i]
            lower_channel_6h = align_htf_to_ltf(prices, df_1w, lower_channel_array)[i]
            ma_1w_6h = align_htf_to_ltf(prices, df_1w, ma_1w_array)[i]
            
            # Volume filter: current volume > 1.4x 5-period average
            vol_ma = np.mean(volume[max(0, i-5):i]) if i >= 5 else volume[i]
            
            # Volatility filter: current ATR > 30th percentile of last 50 periods
            vol_filter = True
            if i >= 50:
                vol_percentile = np.percentile(tr[max(0, i-50):i+1], 30)
                vol_filter = atr[i] > vol_percentile
            
            # Trend filter: daily RSI between 35 and 65 (avoid extremes)
            rsi_filter = (rsi_14[i] >= 35) & (rsi_14[i] <= 65)
            
            if position == 0:
                # Long: Price breaks above upper channel with volume, volatility, and RSI filter
                if (close[i] > upper_channel_6h and close[i-1] <= upper_channel_6h and 
                    volume[i] > vol_ma * 1.4 and 
                    close[i] > donchian_high_1w[i] and  # Breakout confirmation
                    close[i] > ma_1w_6h and  # Above weekly MA
                    vol_filter and rsi_filter):
                    position = 1
                    signals[i] = position_size
                # Short: Price breaks below lower channel with volume, volatility, and RSI filter
                elif (close[i] < lower_channel_6h and close[i-1] >= lower_channel_6h and 
                      volume[i] > vol_ma * 1.4 and 
                      close[i] < donchian_low_1w[i] and  # Breakdown confirmation
                      close[i] < ma_1w_6h and  # Below weekly MA
                      vol_filter and rsi_filter):
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price breaks below lower channel or drops below weekly Donchian low
                if close[i] < lower_channel_6h or close[i] < donchian_low_1w[i]:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price breaks above upper channel or rises above weekly Donchian high
                if close[i] > upper_channel_6h or close[i] > donchian_high_1w[i]:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "6h_WK_ATR_Channel_Donchian_Volume_RSI_Filter"
timeframe = "6h"
leverage = 1.0