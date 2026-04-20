#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data once for multiple indicators
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Daily ATR (14) for volatility and stop sizing
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Daily volume average (20)
    vol_ma_daily = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    # Daily RSI (14) for momentum
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_daily = 100 - (100 / (1 + rs))
    rsi_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi_daily)
    
    # Daily KAMA for trend direction
    # Efficiency Ratio
    change = np.abs(np.diff(close_daily, k=10, prepend=close_daily[:10]))
    volatility = np.sum(np.abs(np.diff(close_daily, prepend=close_daily[0])), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close_daily, np.nan)
    kama[9] = close_daily[9]  # seed
    for i in range(10, len(close_daily)):
        kama[i] = kama[i-1] + sc[i] * (close_daily[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    # Weekly EMA (34)
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Main timeframe data (1d)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_weekly_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_daily_aligned[i]) or np.isnan(atr_daily_aligned[i]) or 
            np.isnan(vol_ma_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34_weekly = ema34_weekly_aligned[i]
        kama = kama_aligned[i]
        rsi = rsi_daily_aligned[i]
        atr = atr_daily_aligned[i]
        vol_ma = vol_ma_daily_aligned[i]
        vol_current = volume[i]
        
        # Trend filter: price above weekly EMA34 AND above daily KAMA
        bullish = price > ema34_weekly and price > kama
        bearish = price < ema34_weekly and price < kama
        
        # Momentum filter: RSI not extreme
        mom_ok = 30 < rsi < 70
        
        # Volatility filter: avoid extremely low volatility
        vol_filter_ok = atr > 0
        
        # Volume filter: current volume > 1.3x daily average
        vol_ok = vol_current > 1.3 * vol_ma
        
        if position == 0:
            # Long: bullish trend with volume and momentum
            if bullish and vol_ok and mom_ok and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish trend with volume and momentum
            elif bearish and vol_ok and mom_ok and vol_filter_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend breaks or momentum deteriorates
            if not bullish or rsi > 75 or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend breaks or momentum deteriorates
            if not bearish or rsi < 25 or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_EMA34_RSI_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0