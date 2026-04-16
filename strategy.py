#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA trend filter + volume spike
# Williams %R identifies overbought/oversold conditions that work in both bull/bear markets.
# 1d EMA filter ensures we trade with the higher timeframe trend.
# Volume spike (2.0x 20-period MA) confirms conviction on reversal signals.
# ATR-based stoploss manages risk with proper alignment.
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and fees.

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
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 4h Williams %R (14) ===
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_4h) / (highest_high_14 - lowest_low_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # === 1d EMA Trend Filter (34) ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h Volume Spike (2.0x 20-period MA) ===
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # === 4h ATR (15) for stoploss ===
    tr = np.maximum(high_4h - low_4h, np.maximum(np.abs(high_4h - np.roll(close_4h, 1)), np.abs(low_4h - np.roll(close_4h, 1))))
    tr[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr).ewm(span=15, adjust=False, min_periods=15).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        wr = williams_r_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 2.0  # 2.0x average volume for spike
        atr_val = atr_aligned[i]
        
        # === STOPLOSS LOGIC (2.0*ATR) ===
        if position == 1:  # Long position
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike confirmation
            if vol_confirm:
                # Long when Williams %R crosses above -80 (oversold) AND price above 1d EMA
                if wr > -80 and price > ema_trend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Short when Williams %R crosses below -20 (overbought) AND price below 1d EMA
                elif wr < -20 and price < ema_trend:
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

name = "4h_WilliamsR_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0