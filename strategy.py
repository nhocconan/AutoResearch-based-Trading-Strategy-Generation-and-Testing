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
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w data (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Calculate ATR on 12h for stoploss ===
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === Calculate 1d ATR for volatility filter ===
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === Calculate 1w ATR for volatility filter ===
    tr_1w = np.maximum(high_1w - low_1w,
                       np.maximum(np.abs(high_1w - np.roll(close_1w, 1)),
                                  np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === Calculate 1d EMA(34) for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Calculate 1w EMA(34) for trend filter ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Calculate 12h RSI(14) for momentum filter ===
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # === Calculate 12h volume ratio (current volume / 20-period average) ===
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_12h / vol_ma_20
    vol_ratio_aligned = vol_ratio  # Already on 12h timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(rsi_14[i]) or
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_12h[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        atr_12h_val = atr_12h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_1w_val = atr_1w_aligned[i]
        rsi_val = rsi_14[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 1d EMA(34) OR RSI > 70 (overbought) OR stoploss hit
            if (price < ema_34_1d_val) or (rsi_val > 70) or (price <= entry_price - 2.0 * atr_12h_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 1d EMA(34) OR RSI < 30 (oversold) OR stoploss hit
            if (price > ema_34_1d_val) or (rsi_val < 30) or (price >= entry_price + 2.0 * atr_12h_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volatility filter: only trade when volatility is elevated (above average)
            vol_filter = (atr_1d_val > np.nanmedian(atr_1d_aligned[max(0, i-50):i]) and 
                         atr_1w_val > np.nanmedian(atr_1w_aligned[max(0, i-50):i]))
            
            # LONG: Price above both 1d and 1w EMA(34) AND RSI between 40 and 60 AND volume surge
            if (price > ema_34_1d_val and price > ema_34_1w_val and 
                40 <= rsi_val <= 60 and vol_ratio_val > 1.5 and vol_filter):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            
            # SHORT: Price below both 1d and 1w EMA(34) AND RSI between 40 and 60 AND volume surge
            elif (price < ema_34_1d_val and price < ema_34_1w_val and 
                  40 <= rsi_val <= 60 and vol_ratio_val > 1.5 and vol_filter):
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_EMA34_1w_RSI_Volume_Filter"
timeframe = "12h"
leverage = 1.0