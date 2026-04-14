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
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly ATR for volatility filter (14-period)
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.roll(close_1w, 1))
    low_close = np.abs(low_1w - np.roll(close_1w, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly Donchian channels (20-period) - breakout levels
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate weekly 14-period RSI for trend filter
    delta = np.diff(close_1w, prepend=close_1w[0])
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
        
        # Get previous week's data (1w index)
        if i >= 1:
            prev_close = close_1w[i-1]
            prev_high = high_1w[i-1]
            prev_low = low_1w[i-1]
            
            # Calculate Donchian breakout levels (previous week)
            dh = donchian_high[i-1]
            dl = donchian_low[i-1]
            
            # Align Donchian levels to weekly timeframe (constant values for the week)
            dh_array = np.full(len(df_1w), dh)
            dl_array = np.full(len(df_1w), dl)
            dh_1w = align_htf_to_ltf(prices, df_1w, dh_array)[i]
            dl_1w = align_htf_to_ltf(prices, df_1w, dl_array)[i]
            
            # Volume filter: current volume > 1.3x 5-period average
            vol_ma = np.mean(volume[max(0, i-5):i]) if i >= 5 else volume[i]
            
            # Volatility filter: current ATR > 25th percentile of last 50 periods
            vol_filter = True
            if i >= 50:
                vol_percentile = np.percentile(tr[max(0, i-50):i+1], 25)
                vol_filter = atr[i] > vol_percentile
            
            # Trend filter: weekly RSI between 40 and 60 (avoid extremes)
            rsi_filter = (rsi_14[i] >= 40) & (rsi_14[i] <= 60)
            
            if position == 0:
                # Long: Price breaks above Donchian high with volume, volatility, and RSI filter
                if (close[i] > dh_1w and close[i-1] <= dh_1w and 
                    volume[i] > vol_ma * 1.3 and 
                    vol_filter and rsi_filter):
                    position = 1
                    signals[i] = position_size
                # Short: Price breaks below Donchian low with volume, volatility, and RSI filter
                elif (close[i] < dl_1w and close[i-1] >= dl_1w and 
                      volume[i] > vol_ma * 1.3 and 
                      vol_filter and rsi_filter):
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price breaks below Donchian low (reverse signal)
                if close[i] < dl_1w:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price breaks above Donchian high (reverse signal)
                if close[i] > dh_1w:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "1w_Donchian_Breakout_Volume_RSI_Filter"
timeframe = "1d"
leverage = 1.0