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
    
    # Get 12h data for trend and ATR
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA34 for trend
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 12h ATR14 for volatility filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr12h_aligned = align_htf_to_ltf(prices, df_12h, atr12h)
    
    # Get 1d data for 4h ATR (used in stop loss)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h ATR14 (using 1d data for proper alignment)
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = np.nan
    tr3_1d[0] = np.nan
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 4h Bollinger Bands (20,2) for mean reversion signals
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Calculate 4h volume spike (volume > 1.5x 20-period average)
    vol_ma = close_series.rolling(window=20, min_periods=20).mean().values  # Using close as proxy for volume MA calculation
    vol_ma_actual = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_actual)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # wait for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or
            np.isnan(atr12h_aligned[i]) or
            np.isnan(atr14_1d_aligned[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 12h EMA34
        uptrend = close[i] > ema34_12h_aligned[i]
        downtrend = close[i] < ema34_12h_aligned[i]
        
        # Volatility filter: only trade when volatility is elevated
        vol_filter = atr12h_aligned[i] > 0.5 * np.nanmean(atr12h_aligned[max(0, i-50):i+1]) if i >= 50 else False
        
        if position == 0:
            # Long: price touches lower Bollinger Band with volume spike in uptrend
            if (low[i] <= bb_lower[i] and close[i] > bb_lower[i]) and volume_spike[i] and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper Bollinger Band with volume spike in downtrend
            elif (high[i] >= bb_upper[i] and close[i] < bb_upper[i]) and volume_spike[i] and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches middle Bollinger Band or reverses
            if close[i] >= bb_middle[i] or (high[i] >= bb_upper[i] and close[i] <= bb_upper[i] * 0.999):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches middle Bollinger Band or reverses
            if close[i] <= bb_middle[i] or (low[i] <= bb_lower[i] and close[i] >= bb_lower[i] * 1.001):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BollingerMeanReversion_EMA34Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0