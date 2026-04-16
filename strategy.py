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
    
    # === 4h data (HTF for direction) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4x ATR for stop loss (4h period)
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # === 12h data (HTF for regime) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Supertrend for regime filtering (ATR=10, multiplier=3)
    atr_12h = pd.Series(
        np.maximum(
            np.maximum(high_12h - low_12h,
                      np.abs(high_12h - np.roll(close_12h, 1))),
            np.abs(low_12h - np.roll(close_12h, 1))
        )
    ).rolling(window=10, min_periods=10).mean().values
    atr_12h[0:9] = np.inf  # handle first values
    
    upper_band = ((high_12h + low_12h) / 2) + (3 * atr_12h)
    lower_band = ((high_12h + low_12h) / 2) - (3 * atr_12h)
    
    supertrend = np.ones_like(close_12h)
    for i in range(1, len(close_12h)):
        if close_12h[i] <= upper_band[i-1]:
            supertrend[i] = 1
        elif close_12h[i] >= lower_band[i-1]:
            supertrend[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            if supertrend[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if supertrend[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
    
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    
    # === 1d data (HTF for volume filter) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / (vol_ma_20_1d + 1e-10)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 4h indicators for entry timing ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (4h)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / vol_ma_10
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state and entry price for stop loss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        atr_4h_val = atr_4h_aligned[i]
        supertrend_val = supertrend_aligned[i]
        vol_ratio_1d_val = vol_ratio_1d_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === STOP LOSS LOGIC ===
        if position == 1:  # Long position
            if price < entry_price - (2.0 * atr_4h_val):
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            if price > entry_price + (2.0 * atr_4h_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === EXIT LOGIC (regime change) ===
        if position == 1:  # Long position
            # Exit when supertrend turns bearish
            if supertrend_val == -1:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            # Exit when supertrend turns bullish
            if supertrend_val == 1:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session with strong daily volume
            if in_session and vol_ratio_1d_val > 1.5:
                # LONG: Supertrend bullish AND RSI not overbought AND volume spike
                if (supertrend_val == 1) and (rsi_val < 70) and (vol_ratio_val > 2.0):
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                
                # SHORT: Supertrend bearish AND RSI not oversold AND volume spike
                elif (supertrend_val == -1) and (rsi_val > 30) and (vol_ratio_val > 2.0):
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

name = "4h_Supertrend_RSI_Volume"
timeframe = "4h"
leverage = 1.0