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
    
    # === 4h EMA(34) for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d RSI(14) for overbought/oversold filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # === 1h Volume Spike (2.0x 20-period average) ===
    vol_ma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute hour filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        # Volume confirmation: current 1h volume > 2.0x 20-period average
        vol_confirm = volume[i] > vol_ma_20_1h[i] * 2.0
        
        # RSI filter: avoid extreme overbought/oversold conditions
        rsi_ok = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        # Entry logic: only enter when flat and in session
        if position == 0 and in_session:
            # Long: price above 4h EMA34 with volume confirmation and RSI not overbought
            if close[i] > ema_34_4h_aligned[i] and vol_confirm and rsi_ok:
                signals[i] = 0.20
                position = 1
                continue
            # Short: price below 4h EMA34 with volume confirmation and RSI not oversold
            elif close[i] < ema_34_4h_aligned[i] and vol_confirm and rsi_ok:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic: price crosses back over 4h EMA34
        elif position == 1:
            # Exit long: price crosses below 4h EMA34
            if close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above 4h EMA34
            if close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA34_1dRSI_VolumeSession"
timeframe = "1h"
leverage = 1.0