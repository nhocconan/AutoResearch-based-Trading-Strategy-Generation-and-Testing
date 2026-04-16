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
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 12h data (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR on 12h for trend filter
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === 12h EMA(34) for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 1d volume spike filter (lookback 20) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    vol_spike_threshold = vol_avg_20_aligned * 1.5  # 50% above average
    
    # === 4h RSI(14) for momentum filter ===
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_spike_threshold[i]) or np.isnan(rsi_14[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_4h[i]
        ema_34_val = ema_34_12h_aligned[i]
        atr_12h_val = atr_12h_aligned[i]
        vol_spike = vol_spike_threshold[i]
        rsi_val = rsi_14[i]
        vol_4h = volume_4h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 12h EMA(34) OR RSI > 70 (overbought)
            if (price < ema_34_val) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 12h EMA(34) OR RSI < 30 (oversold)
            if (price > ema_34_val) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Check volume spike condition (current 4h volume > 1.5x 1d average)
            volume_condition = vol_4h > vol_spike
            
            # LONG: Price above 12h EMA(34) AND volume spike AND RSI between 40 and 60
            if (price > ema_34_val) and volume_condition and (40 <= rsi_val <= 60):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price below 12h EMA(34) AND volume spike AND RSI between 40 and 60
            elif (price < ema_34_val) and volume_condition and (40 <= rsi_val <= 60):
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

name = "4h_EMA34_12h_VolumeSpike_RSI_Filter"
timeframe = "4h"
leverage = 1.0