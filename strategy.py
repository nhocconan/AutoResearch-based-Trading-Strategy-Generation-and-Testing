#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 12h data (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # === 4h Donchian channel (20-period) ===
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # === 4h 20-period EMA for trend filter ===
    ema_20 = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 12h RSI (14-period) for momentum filter ===
    delta = pd.Series(close_12h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # === 4h volume ratio (20-period) ===
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_4h / vol_ma_20
    
    # === 4h ATR (14-period) for stop loss ===
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    
    # Track position and entry price for stop loss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: enough for Donchian, EMA, RSI calculations
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema = ema_20_aligned[i]
        rsi = rsi_12h_aligned[i]
        vol = vol_ratio_aligned[i]
        atr_val = atr_aligned[i]
        
        # === STOP LOSS LOGIC ===
        if position == 1:  # Long position
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below EMA OR RSI > 70 (overbought)
            if price < ema or rsi > 70:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            # Exit: price closes above EMA OR RSI < 30 (oversold)
            if price > ema or rsi < 30:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high + EMA uptrend + RSI not overbought + volume
            if price > upper and price > ema and rsi < 70 and vol > 1.3:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Price breaks below Donchian low + EMA downtrend + RSI not oversold + volume
            elif price < lower and price < ema and rsi > 30 and vol > 1.3:
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

name = "4h_Donchian_EMA_RSI_Volume"
timeframe = "4h"
leverage = 1.0