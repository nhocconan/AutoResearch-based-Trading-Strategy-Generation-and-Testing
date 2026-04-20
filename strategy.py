#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily volume ratio (current / 20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma_20_1d == 0, 1, vol_ma_20_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Daily price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channel (20-period)
    donch_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend_1w = ema_21_1w_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio_1d = vol_ratio_1d_aligned[i]
        
        # Trend filter: price above/below weekly EMA
        trend_up = price > ema_trend_1w
        trend_down = price < ema_trend_1w
        
        # Volatility filter: avoid extreme volatility
        atr_ma_10 = pd.Series(atr_14_1d_aligned).rolling(window=10, min_periods=10).mean().values[i]
        vol_filter = (atr > 0.3 * atr_ma_10) and (atr < 4.0 * atr_ma_10)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_1d > 1.2)
        
        if position == 0:
            # Enter long on Donchian breakout with trend and volume confirmation
            if trend_up and vol_filter and (price > donch_high[i]):
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian breakdown with trend and volume confirmation
            elif trend_down and vol_filter and (price < donch_low[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or volatility spike
            if (not trend_up) or (atr > 3.5 * atr_ma_10):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or volatility spike
            if (not trend_down) or (atr > 3.5 * atr_ma_10):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA_DonchianBreakout_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0