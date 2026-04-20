#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data ONCE for HTF regime
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily ATR(14) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA(50) for trend regime
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Daily ATR(20) for Donchian width
    tr_atr20 = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20_1d = pd.Series(tr_atr20).rolling(window=20, min_periods=20).mean().values
    
    # Daily Donchian channels (20-period)
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily indicators to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in HTF indicators
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_20_1d_aligned[i]) or np.isnan(high_20_1d_aligned[i]) or np.isnan(low_20_1d_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_1d_aligned[i]
        atr_14_val = atr_14_1d_aligned[i]
        atr_20_val = atr_20_1d_aligned[i]
        high_20_val = high_20_1d_aligned[i]
        low_20_val = low_20_1d_aligned[i]
        vol_ma_val = volume_ma_20_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume must be above 20-day average
        vol_filter = vol > vol_ma_val
        
        # Volatility regime: only trade when volatility is below median (calm markets)
        vol_regime = atr_14_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 50)
        
        if position == 0:
            # Long: price breaks above daily Donchian high (uptrend breakout), volume confirmation, calm volatility
            if price > high_20_val and vol_filter and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low (downtrend breakout), volume confirmation, calm volatility
            elif price < low_20_val and vol_filter and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA50 or volatility spikes or Donchian mean reversion
            if price < ema_50_val or atr_14_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 70) or price < (high_20_val + low_20_val) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA50 or volatility spikes or Donchian mean reversion
            if price > ema_50_val or atr_14_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 70) or price > (high_20_val + low_20_val) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeVolatilityFilter_V1"
timeframe = "12h"
leverage = 1.0