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
    
    # Load 12-hour data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h RSI for overbought/oversold
    delta = pd.Series(close_12h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h.values)
    
    # Calculate 4-hour Donchian channels (20-period) for breakout signals
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    for i in range(20, n):
        donchian_high[i] = high_series.iloc[i-20:i].max()
        donchian_low[i] = low_series.iloc[i-20:i].min()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and trend confirmation
            vol_ma = np.mean(volume[i-5:i]) if i >= 5 else volume[i]
            if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i] and 
                volume[i] > vol_ma * 1.5 and 
                close[i] > ema50_12h_aligned[i] and 
                rsi_12h_aligned[i] < 70):  # Not overbought
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian low with volume and trend confirmation
            elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i] and 
                  volume[i] > vol_ma * 1.5 and 
                  close[i] < ema50_12h_aligned[i] and 
                  rsi_12h_aligned[i] > 30):  # Not oversold
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below Donchian low or trend reversal (price below EMA50)
            if close[i] < donchian_low[i] or close[i] < ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above Donchian high or trend reversal (price above EMA50)
            if close[i] > donchian_high[i] or close[i] > ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_EMA50_RSI_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0