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
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate ATR on 12h
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === 1w data (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly moving average
    ma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ma_20_1w)
    
    # Calculate weekly volume average
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    # === 12h EMA(34) for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 12h RSI(14) for momentum filter ===
    delta = np.diff(close_12h, prepend=close_12h[0])
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
            np.isnan(ma_20_1w_aligned[i]) or np.isnan(vol_avg_20_1w_aligned[i]) or 
            np.isnan(rsi_14[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        ema_34_val = ema_34_12h_aligned[i]
        atr_12h_val = atr_12h_aligned[i]
        ma_20_1w_val = ma_20_1w_aligned[i]
        vol_avg_20_1w_val = vol_avg_20_1w_aligned[i]
        rsi_val = rsi_14[i]
        
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
            # LONG: Price above 12h EMA(34) AND RSI between 40 and 60 (neutral momentum)
            # AND weekly trend is up (price > 20w MA) AND volume above average
            if (price > ema_34_val) and (40 <= rsi_val <= 60) and \
               (price > ma_20_1w_val) and (vol > vol_avg_20_1w_val):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price below 12h EMA(34) AND RSI between 40 and 60 (neutral momentum)
            # AND weekly trend is down (price < 20w MA) AND volume above average
            elif (price < ema_34_val) and (40 <= rsi_val <= 60) and \
                 (price < ma_20_1w_val) and (vol > vol_avg_20_1w_val):
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

name = "12h_EMA34_1wTrend_RSI_VolumeFilter"
timeframe = "12h"
leverage = 1.0