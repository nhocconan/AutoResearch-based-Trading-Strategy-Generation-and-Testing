#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Calculate weekly KAMA (adaptive moving average) for trend direction
    # Efficiency ratio over 10 periods
    change_10 = np.abs(np.diff(weekly_close, n=10, prepend=weekly_close[0]))
    volatility_10 = np.sum(np.abs(np.diff(weekly_close, prepend=weekly_close[0])), axis=0) if len(weekly_close.shape) > 1 else \
                  np.sum(np.abs(np.diff(weekly_close, prepend=weekly_close[0])))
    if len(weekly_close.shape) == 1:
        volatility_10 = pd.Series(weekly_close).diff().abs().rolling(window=10, min_periods=1).sum().values
    er = change_10 / (volatility_10 + 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(weekly_close)
    kama[0] = weekly_close[0]
    for i in range(1, len(weekly_close)):
        kama[i] = kama[i-1] + sc[i] * (weekly_close[i] - kama[i-1])
    
    # Calculate weekly RSI(14) for overbought/oversold
    delta = pd.Series(weekly_close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    weekly_rsi = 100 - (100 / (1 + rs))
    weekly_rsi = weekly_rsi.values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = pd.Series(weekly_high - weekly_low)
    tr2 = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3 = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    weekly_atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1d timeframe with proper delay
    kama_1d = align_htf_to_ltf(prices, df_1w, kama)
    weekly_rsi_1d = align_htf_to_ltf(prices, df_1w, weekly_rsi)
    weekly_atr_1d = align_htf_to_ltf(prices, df_1w, weekly_atr)
    
    # Calculate 1d Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d[i]) or np.isnan(weekly_rsi_1d[i]) or np.isnan(weekly_atr_1d[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly KAMA trend filter: price above/below KAMA
        # 2. Weekly RSI extreme (oversold/overbought) for mean reversion
        # 3. 1d Donchian breakout in direction of RSI signal
        # 4. 1d volume confirmation: volume > 2.0x average
        # 5. 1d volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 6. Discrete position sizing: 0.25
        
        # Long conditions: Weekly RSI oversold (< 30) + Donchian breakout above in uptrend
        if (weekly_rsi_1d[i] < 30 and          # Weekly oversold
            close[i] > kama_1d[i] and          # Weekly uptrend filter
            close[i] > highest_20[i] and       # 1d Donchian breakout
            volume_ratio[i] > 2.0 and          # Volume confirmation
            weekly_atr_1d[i] > 0.005 * close[i]):  # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: Weekly RSI overbought (> 70) + Donchian breakdown below in downtrend
        elif (weekly_rsi_1d[i] > 70 and        # Weekly overbought
              close[i] < kama_1d[i] and        # Weekly downtrend filter
              close[i] < lowest_20[i] and      # 1d Donchian breakdown
              volume_ratio[i] > 2.0 and        # Volume confirmation
              weekly_atr_1d[i] > 0.005 * close[i]):  # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyKAMA_RSI_Donchian_Breakout_Volume_ATR_Filter"
timeframe = "1d"
leverage = 1.0