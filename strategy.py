#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Daily data for ATR-based volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR (14-period) for volatility regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_14d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 12h ATR for entry/exit
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h EMA50 for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20
    
    # 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(atr_14d[i]) or np.isnan(atr_12h[i]) or np.isnan(ema_50[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: only trade when volatility is elevated (above median)
        vol_regime = atr_14d[i] > np.nanmedian(atr_14d[max(0, i-100):i+1])
        
        if position == 0 and vol_regime:
            # Long: Price breaks above Donchian high with volume surge and above EMA50
            if close[i] > highest_high[i] and vol_surge[i] and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume surge and below EMA50
            elif close[i] < lowest_low[i] and vol_surge[i] and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or volatility drops
            if position == 1:
                if close[i] < lowest_low[i] or not vol_regime:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > highest_high[i] or not vol_regime:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_VolRegime_VolSurge_v1"
timeframe = "12h"
leverage = 1.0