#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Trend filter (EMA 200) ===
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 4h: Donchian channel (20 periods) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian upper and lower bands
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # ATR for volatility filter (14 periods)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        ema_200_val = ema_200_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or 
            np.isnan(vol_ratio_val) or np.isnan(atr_val) or np.isnan(ema_200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median
        atr_median = np.nanmedian(atr[max(0, i-49):i+1]) if i >= 1 else np.nan
        vol_filter = atr_val > atr_median if not np.isnan(atr_median) else False
        
        # Trend filter: only long when price > EMA200, short when price < EMA200
        trend_filter_long = close_val > ema_200_val
        trend_filter_short = close_val < ema_200_val
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume confirmation, volatility filter, and trend filter
            if (close_val > donchian_high_val and   # Break above Donchian high
                vol_ratio_val > 2.0 and             # Strong volume confirmation
                vol_filter and                      # Volatility filter
                trend_filter_long):                 # Trend filter (bullish)
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume confirmation, volatility filter, and trend filter
            elif (close_val < donchian_low_val and  # Break below Donchian low
                  vol_ratio_val > 2.0 and           # Strong volume confirmation
                  vol_filter and                    # Volatility filter
                  trend_filter_short):              # Trend filter (bearish)
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below Donchian low (reversion to mean)
            if close_val < donchian_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above Donchian high (reversion to mean)
            if close_val > donchian_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals