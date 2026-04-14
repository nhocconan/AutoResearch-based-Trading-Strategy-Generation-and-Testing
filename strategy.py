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
    
    # Calculate 4-hour ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4-hour Donchian channels (20-period) - breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate daily 14-period RSI for trend filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate daily ATR for volatility regime filter
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    high_close_1d[0] = high_low_1d[0]
    low_close_1d[0] = high_low_1d[0]
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    tr_series_1d = pd.Series(tr_1d)
    atr_1d = tr_series_1d.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(atr[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(rsi_14[i]) or np.isnan(atr_1d[i]):
            continue
        
        # Get previous day's data (1d index)
        if i >= 1:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            
            # Calculate volatility regime: current daily ATR > 50th percentile of last 20 days
            vol_regime = True
            if i >= 20:
                vol_percentile = np.percentile(atr_1d[max(0, i-20):i+1], 50)
                vol_regime = atr_1d[i] > vol_percentile
            
            # Trend filter: daily RSI between 40 and 60 (avoid extremes)
            rsi_filter = (rsi_14[i] >= 40) & (rsi_14[i] <= 60)
            
            if position == 0:
                # Long: Price breaks above Donchian high with volume, volatility regime, and RSI filter
                if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i] and 
                    volume[i] > np.mean(volume[max(0, i-5):i]) * 1.5 if i >= 5 else volume[i] > 0 and
                    vol_regime and rsi_filter):
                    position = 1
                    signals[i] = position_size
                # Short: Price breaks below Donchian low with volume, volatility regime, and RSI filter
                elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i] and 
                      volume[i] > np.mean(volume[max(0, i-5):i]) * 1.5 if i >= 5 else volume[i] > 0 and
                      vol_regime and rsi_filter):
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price breaks below Donchian low (reverse signal)
                if close[i] < donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price breaks above Donchian high (reverse signal)
                if close[i] > donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_Regime_RSI_Filter"
timeframe = "4h"
leverage = 1.0