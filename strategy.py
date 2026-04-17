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
    
    # Get daily data for 14-period RSI (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily RSI14 for trend filter
    close_1d_series = pd.Series(close_1d)
    delta = close_1d_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi14_1d = (100 - (100 / (1 + rs))).values
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    # Get 12h data for Donchian channel (price channel)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian(20) channel
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    donch_high_20 = high_12h_series.rolling(window=20, min_periods=20).max().values
    donch_low_20 = low_12h_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_12h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_12h, donch_low_20)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # ATR(14) for volatility filter and stop
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi14_1d_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian high with volume and daily RSI > 50
            if close[i] > donch_high_20_aligned[i] and volume_filter[i] and rsi14_1d_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low with volume and daily RSI < 50
            elif close[i] < donch_low_20_aligned[i] and volume_filter[i] and rsi14_1d_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 12h Donchian low OR ATR-based stop
            if close[i] < donch_low_20_aligned[i] or close[i] < (high[max(0, i-1)] - 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 12h Donchian high OR ATR-based stop
            if close[i] > donch_high_20_aligned[i] or close[i] > (low[max(0, i-1)] + 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_12h_DailyRSI50_Trend_Volume"
timeframe = "12h"
leverage = 1.0