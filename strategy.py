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
    
    # === Daily ATR for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === Weekly EMA34 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 14-day RSI for momentum ===
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain_ma.values / np.maximum(loss_ma.values, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume Spike Detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200  # Need RSI(14), EMA34(1w), ATR14
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema34w = ema_34_1w_aligned[i]
        atr = atr_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Exit when trend weakens or volatility drops ===
        if position == 1:  # Long position
            # Exit when RSI turns weak OR volatility drops significantly
            if rsi_val < 50 or atr < (atr_1d_aligned[i-1] * 0.6 if i > 0 else atr):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when RSI turns strong OR volatility drops significantly
            if rsi_val > 50 or atr < (atr_1d_aligned[i-1] * 0.6 if i > 0 else atr):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI > 60, price above weekly EMA34, volume spike
            if rsi_val > 60 and price > ema34w and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: RSI < 40, price below weekly EMA34, volume spike
            elif rsi_val < 40 and price < ema34w and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_RSI14_EMA34_1w_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0