#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Daily volume spike filter (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d / vol_ma_1d
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 4h RSI for mean reversion entry
    close_4h = prices['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(140, n):
        # Skip if indicators not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close_4h[i]
        ema50 = ema50_12h_aligned[i]
        vol_spike = vol_spike_1d_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: price above 12h EMA50, oversold RSI, volume spike
            if (price_close > ema50 and 
                rsi_val < 30 and 
                vol_spike > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: price below 12h EMA50, overbought RSI, volume spike
            elif (price_close < ema50 and 
                  rsi_val > 70 and 
                  vol_spike > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: RSI mean reversion or trend change
            if position == 1 and (rsi_val > 70 or price_close < ema50):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi_val < 30 or price_close > ema50):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_12hEMA50_RSI_MeanReversion_VolumeSpike"
timeframe = "4h"
leverage = 1.0