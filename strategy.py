#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (HTF for trend and key levels) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 4h KAMA trend (ER=10) ===
    change_4h = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    vol_4h = np.sum(np.abs(np.diff(close_4h, prepend=close_4h[0])), 
                    axis=0) if len(close_4h) > 1 else np.zeros_like(close_4h)
    er_4h = np.where(vol_4h != 0, change_4h / vol_4h, 0)
    sc_4h = (er_4h * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama_4h = np.zeros_like(close_4h)
    kama_4h[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama_4h[i] = kama_4h[i-1] + sc_4h[i] * (close_4h[i] - kama_4h[i-1])
    kama_4h = kama_4h.astype(np.float64)
    
    # === 4h ATR for volatility filter ===
    tr_4h = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.maximum(
            np.abs(high_4h[1:] - close_4h[:-1]),
            np.abs(low_4h[1:] - close_4h[:-1])
        )
    )
    tr_4h = np.concatenate([[high_4h[0] - low_4h[0]], tr_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # === 4h Previous Day Range (for Keltner-like channels) ===
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h[0] = close_4h[0]
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    
    prev_range_4h = prev_high_4h - prev_low_4h
    k_mult = 1.5
    
    # === Keltner Channel middle (EMA20) ===
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Upper and Lower Bands ===
    upper_4h = ema_20_4h + k_mult * atr_4h
    lower_4h = ema_20_4h - k_mult * atr_4h
    
    # Align HTF indicators to 1h
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # === Volume confirmation (1h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN or outside session
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(upper_4h_aligned[i]) or 
            np.isnan(lower_4h_aligned[i]) or np.isnan(vol_ratio[i]) or
            hours[i] < 8 or hours[i] > 20):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        kama_val = kama_4h_aligned[i]
        upper_val = upper_4h_aligned[i]
        lower_val = lower_4h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below lower band (stop) or hits upper band (take profit)
            if price < lower_val or price > upper_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above upper band (stop) or hits lower band (take profit)
            if price > upper_val or price < lower_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper band with volume AND above KAMA (uptrend)
            if (price > upper_val) and (price > kama_val) and (vol_ratio_val > 1.8):
                signals[i] = 0.20
                position = 1
                continue
            
            # SHORT: Price breaks below lower band with volume AND below KAMA (downtrend)
            elif (price < lower_val) and (price < kama_val) and (vol_ratio_val > 1.8):
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

name = "1h_KAMA_Keltner_Breakout_Volume_SessionFilter"
timeframe = "1h"
leverage = 1.0