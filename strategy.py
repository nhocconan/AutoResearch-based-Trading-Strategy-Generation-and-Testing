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
    
    # Get weekly data for regime and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly ATR(14) for volatility filter
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    tr_weekly = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
    tr_weekly = tr_weekly.max(axis=1)
    atr14_1w = tr_weekly.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    # Get daily data for price action
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian(20) for breakout levels
    donch_high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    donch_low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Daily volume average for confirmation
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(atr14_1w_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1w_aligned[i]
        atr_vol = atr14_1w_aligned[i]
        donch_high = donch_high_20_aligned[i]
        donch_low = donch_low_20_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        vol_ratio = df_1d['volume'].iloc[i] / vol_ma if vol_ma > 0 else 0
        
        if position == 0:
            # Long: break above Donchian high with volume surge in weekly uptrend
            if (high[i] > donch_high and 
                close[i] > donch_high and 
                ema_trend > close[i] * 0.98 and  # Price above weekly EMA (uptrend)
                vol_ratio > 2.0):  # Volume at least 2x average
                signals[i] = size
                position = 1
            # Short: break below Donchian low with volume surge in weekly downtrend
            elif (low[i] < donch_low and 
                  close[i] < donch_low and 
                  ema_trend < close[i] * 1.02 and  # Price below weekly EMA (downtrend)
                  vol_ratio > 2.0):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low or trend weakens
            if low[i] <= donch_low or ema_trend < close[i] * 0.95:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high or trend weakens
            if high[i] >= donch_high or ema_trend > close[i] * 1.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSurge_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0