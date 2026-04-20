#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_PriceAction_TrendV2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d: Price action and momentum ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 14-period RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # 20-period SMA
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 1w: Trend filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 21-period EMA for weekly trend
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align HTF indicators to 4h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_14_aligned[i]
        sma_val = sma_20_aligned[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        ema_1w_val = ema_21_1w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(sma_val) or 
            np.isnan(vol_ratio_val) or np.isnan(ema_1w_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish weekly trend, price above SMA, RSI not overbought, volume confirmation
            if (close_val > ema_1w_val and      # Above weekly EMA (bullish trend)
                close_val > sma_val and         # Above daily SMA
                rsi_val < 70 and                # Not overbought
                vol_ratio_val > 1.5):           # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Bearish weekly trend, price below SMA, RSI not oversold, volume confirmation
            elif (close_val < ema_1w_val and    # Below weekly EMA (bearish trend)
                  close_val < sma_val and       # Below daily SMA
                  rsi_val > 30 and              # Not oversold
                  vol_ratio_val > 1.5):         # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below SMA or weekly trend turns bearish
            if close_val < sma_val or close_val < ema_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above SMA or weekly trend turns bullish
            if close_val > sma_val or close_val > ema_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals