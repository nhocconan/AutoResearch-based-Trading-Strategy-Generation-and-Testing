#!/usr/bin/env python3
"""
6h_price_channel_rsi_1d_trend_volume_v1
Hypothesis: 6h RSI pullback in direction of 1d trend with volume confirmation works in both bull and bear markets.
- 1d trend: price above/below 100-period EMA
- 6h entry: RSI pullback to 40-60 range with volume confirmation
- Risk management: exit on RSI reversal or trend change
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_price_channel_rsi_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(100) for trend
    close_1d_series = pd.Series(close_1d)
    ema_100 = close_1d_series.ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100)
    
    # 6h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(loss_ma != 0, gain_ma / loss_ma, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if data not available
        if (np.isnan(ema_100_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend turns bearish
            if rsi[i] > 70 or close[i] < ema_100_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend turns bullish
            if rsi[i] < 30 or close[i] > ema_100_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            vol_confirmed = volume[i] > vol_ma[i]
            
            if not vol_confirmed:
                signals[i] = 0.0
                continue
            
            # Long: uptrend + RSI pullback to 40-60
            if close[i] > ema_100_aligned[i] and 40 <= rsi[i] <= 60:
                # Additional check: price closing above open (bullish candle)
                if close[i] > open_price[i]:
                    position = 1
                    signals[i] = 0.25
            # Short: downtrend + RSI pullback to 40-60
            elif close[i] < ema_100_aligned[i] and 40 <= rsi[i] <= 60:
                # Additional check: price closing below open (bearish candle)
                if close[i] < open_price[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals