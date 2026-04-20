#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (primary HTF)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h ADX for trend strength filter
    # Calculate True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = high_4h[0] - low_4h[0]
    tr2[0] = np.abs(high_4h[0] - close_4h[0])
    tr3[0] = np.abs(low_4h[0] - close_4h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DM and -DM
    up_move = np.diff(high_4h, prepend=high_4h[0])
    down_move = np.diff(low_4h, prepend=low_4h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth +DM, -DM, and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_4h
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_4h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_4h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Load daily data for regime filter (secondary HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for long-term trend
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Daily ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = high_1d[0] - low_1d[0]
    tr2_1d[0] = np.abs(high_1d[0] - close_1d[0])
    tr3_1d[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = prices['volume'].iloc[i]
        
        # Trend alignment: 4h EMA50 direction
        trend_up = price > ema_50_4h_aligned[i]
        trend_down = price < ema_50_4h_aligned[i]
        
        # Trend strength: ADX > 25
        strong_trend = adx_4h_aligned[i] > 25
        
        # Long-term trend filter: price vs daily EMA200
        above_long_term = price > ema_200_1d_aligned[i]
        below_long_term = price < ema_200_1d_aligned[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_normal = atr_1d_aligned[i] > 0.5 * np.nanmedian(atr_1d_aligned[max(0, i-100):i+1])
        
        # Volume confirmation: above average volume
        vol_surge = vol > 1.2 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: uptrend + strong trend + above long-term MA + volume surge
            if (trend_up and strong_trend and above_long_term and vol_surge and vol_normal):
                signals[i] = 0.20
                position = 1
            # Short: downtrend + strong trend + below long-term MA + volume surge
            elif (trend_down and strong_trend and below_long_term and vol_surge and vol_normal):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend breaks or ADX weakens
            if not (trend_up and strong_trend) or price < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend breaks or ADX weakens
            if not (trend_down and strong_trend) or price > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hADX_EMA50_VolumeTrend_Filter"
timeframe = "1h"
leverage = 1.0