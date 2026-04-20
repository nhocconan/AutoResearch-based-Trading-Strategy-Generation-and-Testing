#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_TripleTimeframe_Confluence"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 12h EMA Trend Filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_trend_12h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 1d Donchian Channel Breakout Levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # === 4h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma20 > 0, volume / vol_ma20, 0)
    
    # === 4h RSI for Entry Timing ===
    close_series = pd.Series(prices['close'].values)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema_trend_val = ema_trend_12h[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        rsi_val = rsi_values[i]
        
        # Skip if any value is invalid
        if (np.isnan(ema_trend_val) or np.isnan(donchian_high_val) or 
            np.isnan(donchian_low_val) or np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above 12h EMA trend + breaks 1d Donchian high + volume spike + RSI not overbought
            if (close_val > ema_trend_val and 
                close_val > donchian_high_val and 
                vol_ratio_val > 2.0 and 
                rsi_val < 70):
                signals[i] = 0.25
                position = 1
            # Short: Price below 12h EMA trend + breaks 1d Donchian low + volume spike + RSI not oversold
            elif (close_val < ema_trend_val and 
                  close_val < donchian_low_val and 
                  vol_ratio_val > 2.0 and 
                  rsi_val > 30):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below 12h EMA OR RSI overbought
            if close_val < ema_trend_val or rsi_val > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above 12h EMA OR RSI oversold
            if close_val > ema_trend_val or rsi_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals