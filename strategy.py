#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h/1d multi-timeframe confirmation using RSI mean reversion
# In both bull and bear markets, RSI extremes combined with trend filters and volume provide edge
# Uses 4h RSI for direction, 1d trend filter for regime, and 1h for precise entry timing
# Target: 15-37 trades/year with strict entry conditions to avoid fee drag

name = "1h_RSI_MeanReversion_4D_TrendFilter_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h RSI (14-period)
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # Align 4h RSI to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA (50-period) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi = rsi_4h_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: RSI oversold (<30) + price above 1d EMA50 (uptrend filter) + volume
            if rsi < 30 and price > ema_50 and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) + price below 1d EMA50 (downtrend filter) + volume
            elif rsi > 70 and price < ema_50 and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral (>50) or price breaks below 1d EMA50
            if rsi > 50 or price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: RSI returns to neutral (<50) or price breaks above 1d EMA50
            if rsi < 50 or price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals