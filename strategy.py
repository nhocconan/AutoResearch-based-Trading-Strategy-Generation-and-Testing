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
    
    # === Daily data for 1d indicators ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === ATR(14) for volatility filter ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_avg = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # === 4h EMA34 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 4h RSI(14) for momentum filter ===
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_4h = 100 - (100 / (1 + rs))
    
    # === Align HTF indicators to 1h timeframe ===
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    atr_1d_avg_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_avg)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi_14_4h_aligned[i]) or
            np.isnan(atr_1d_avg_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_34 = ema_34_4h_aligned[i]
        rsi_14 = rsi_14_4h_aligned[i]
        atr_avg = atr_1d_avg_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Exit when trend or momentum deteriorates ===
        if position == 1:  # Long position
            if price < ema_34 or rsi_14 < 40:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            if price > ema_34 or rsi_14 > 60:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above EMA34, RSI > 50, volume spike, sufficient volatility
            if price > ema_34 and rsi_14 > 50 and vol_spike and atr_avg > 0:
                signals[i] = 0.20
                position = 1
                continue
            
            # SHORT: Price below EMA34, RSI < 50, volume spike, sufficient volatility
            elif price < ema_34 and rsi_14 < 50 and vol_spike and atr_avg > 0:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_EMA34_RSI14_VolumeSpike_ATRFilter_v1"
timeframe = "1h"
leverage = 1.0