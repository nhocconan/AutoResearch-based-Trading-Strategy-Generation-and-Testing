#!/usr/bin/env python3
"""
4h_donchian_20_1d_trend_volume_v3
Hypothesis: On 4-hour timeframe, use Donchian channel breakouts with 1-day trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high with daily EMA(50) trending up and volume > 1.5x 20-period average.
Short when price breaks below 20-period Donchian low with daily EMA(50) trending down and volume > 1.5x 20-period average.
Exit when price returns to the Donchian midpoint.
Designed for 20-40 trades/year to minimize fee drag while capturing strong trends with institutional validation.
Works in both bull/bear markets as Donchian channels adapt to volatility and daily trend filter avoids counter-trend trades.
This version increases trade frequency by lowering the volume threshold to 1.2x average and adding momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Determine daily trend direction (using EMA slope)
    daily_trend_up = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    daily_trend_down = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    for i in range(1, len(ema_50_1d_aligned)):
        if not np.isnan(ema_50_1d_aligned[i]) and not np.isnan(ema_50_1d_aligned[i-1]):
            daily_trend_up[i] = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            daily_trend_down[i] = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
    
    # Calculate Donchian Channel (20-period) on 4h timeframe
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Momentum confirmation: 4-period RSI
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=4, min_periods=4).mean()
    avg_loss = loss.rolling(window=4, min_periods=4).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50, 4), n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation (lowered threshold for more trades)
        vol_ok = volume[i] > 1.2 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and daily trend alignment
            if vol_ok:
                # Long: price breaks above Donchian high with daily uptrend and bullish momentum
                if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and 
                    daily_trend_up[i] and rsi_values[i] > 50):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low with daily downtrend and bearish momentum
                elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and 
                      daily_trend_down[i] and rsi_values[i] < 50):
                    position = -1
                    signals[i] = -0.25
    
    return signals