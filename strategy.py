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
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1d ADX (14-period)
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > -low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((-low_diff > high_diff) & (-low_diff > 0), -low_diff, 0)
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    tr_series = pd.Series(tr)
    atr_14 = tr_series.ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate 6-day RSI for additional filter
    delta_6d = np.diff(close_1d, prepend=close_1d[0])
    gain_6d = np.where(delta_6d > 0, delta_6d, 0)
    loss_6d = np.where(delta_6d < 0, -delta_6d, 0)
    gain_series_6d = pd.Series(gain_6d)
    loss_series_6d = pd.Series(loss_6d)
    avg_gain_6d = gain_series_6d.ewm(alpha=1/6, adjust=False).mean().values
    avg_loss_6d = loss_series_6d.ewm(alpha=1/6, adjust=False).mean().values
    rs_6d = avg_gain_6d / (avg_loss_6d + 1e-10)
    rsi_6d = 100 - (100 / (1 + rs_6d))
    
    # Calculate 6h volume filter: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 6h Donchian channels (20-period) - breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if np.isnan(rsi_1d[i]) or np.isnan(adx[i]) or np.isnan(rsi_6d[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma[i]):
            continue
        
        # Get previous day's RSI and ADX
        if i >= 1:
            rsi_prev = rsi_1d[i-1]
            adx_prev = adx[i-1]
            rsi_6d_prev = rsi_6d[i-1]
            
            if position == 0:
                # Long: RSI oversold (<30) + ADX rising (>25) + 6-day RSI not overbought
                if (rsi_prev < 30 and adx_prev > 25 and rsi_6d_prev < 70 and
                    volume[i] > vol_ma[i] * 1.5):
                    position = 1
                    signals[i] = position_size
                # Short: RSI overbought (>70) + ADX rising (>25) + 6-day RSI not oversold
                elif (rsi_prev > 70 and adx_prev > 25 and rsi_6d_prev > 30 and
                      volume[i] > vol_ma[i] * 1.5):
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: RSI overbought (>70) or ADX weakening (<20)
                if rsi_1d[i] > 70 or adx[i] < 20:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: RSI oversold (<30) or ADX weakening (<20)
                if rsi_1d[i] < 30 or adx[i] < 20:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "6h_1d_RSI_ADX_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0