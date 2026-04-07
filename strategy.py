#!/usr/bin/env python3
"""
1h_cdl_engulfing_4h1d_trend_volume_v1
Hypothesis: On 1-hour timeframe, trade bullish/bearish engulfing candles with 4h trend filter and 1d volume confirmation.
Long when bullish engulfing forms above 4h EMA(50) with 1d volume > 1.3x 20-day average.
Short when bearish engulfing forms below 4h EMA(50) with 1d volume > 1.3x 20-day average.
Exit on opposite engulfing signal or when price crosses 4h EMA(50) in opposite direction.
Designed for 15-35 trades/year (~60-140 total over 4 years) to minimize fee drag while capturing high-probability reversals.
Works in bull/bear markets as engulfing patterns signal exhaustion and 4h trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_cdl_engulfing_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d 20-period average volume
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Detect engulfing candles
    # Bullish engulfing: current candle engulfs previous bearish candle
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > open_price[1]) & (open_price < close[1]) & \
                     (close[1] < open_price[1])  # previous candle bearish
    
    # Bearish engulfing: current candle engulfs previous bullish candle
    bearish_engulf = (close < open_price) & (open_price > close) & \
                     (close < open_price[1]) & (open_price > close[1]) & \
                     (open_price[1] > close[1])  # previous candle bullish
    
    # Handle first element
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current 1h volume > 1.3x 1d average volume
        # Note: 1d volume is daily, so we compare 1h volume to scaled daily average
        vol_ok = volume[i] > 1.3 * (vol_ma_20_1d_aligned[i] / 24)  # approx 24 hours in a day
        
        if position == 1:  # Long position
            # Exit: bearish engulfing forms OR price crosses below 4h EMA(50)
            if bearish_engulf[i] or (close[i] < ema_50_4h_aligned[i] and close[i-1] >= ema_50_4h_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: bullish engulfing forms OR price crosses above 4h EMA(50)
            if bullish_engulf[i] or (close[i] > ema_50_4h_aligned[i] and close[i-1] <= ema_50_4h_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only enter with volume confirmation and 4h trend alignment
            if vol_ok:
                # Long: bullish engulfing with price above 4h EMA(50) (uptrend)
                if bullish_engulf[i] and close[i] > ema_50_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: bearish engulfing with price below 4h EMA(50) (downtrend)
                elif bearish_engulf[i] and close[i] < ema_50_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals