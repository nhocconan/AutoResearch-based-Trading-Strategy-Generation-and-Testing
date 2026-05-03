#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when: price breaks above 12h Camarilla R3 level AND close > 1d EMA34 AND volume > 2.0x 24-bar average
# Short when: price breaks below 12h Camarilla S3 level AND close < 1d EMA34 AND volume > 2.0x 24-bar average
# Exit via ATR(24) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 12h Camarilla for structure (proven edge from top performers), 1d EMA34 for HTF trend alignment, volume spike for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 50-150 total trades over 4 years = 12-37/year.

name = "12h_Camarilla_R3_S3_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla pivots (based on previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Previous 12h bar
    prev_high = df_12h['high'].iloc[-2]
    prev_low = df_12h['low'].iloc[-2]
    prev_close = df_12h['close'].iloc[-2]
    
    # Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align to LTF (12h) - values from previous completed 12h bar
    R3_aligned = align_htf_to_ltf(prices, df_12h, np.full(len(df_12h), R3))
    S3_aligned = align_htf_to_ltf(prices, df_12h, np.full(len(df_12h), S3))
    
    # Calculate 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 24-bar average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=1).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])]], np.maximum(tr1, np.maximum(tr2, tr3)))
    atr = pd.Series(tr).rolling(window=24, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(24, n):  # Start after warmup for ATR/volume
        # Get current values
        price = close[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ok = vol_spike[i]
        atr_val = atr[i]
        
        # Long entry conditions
        long_entry = (price > r3) and (price > ema_trend) and vol_ok
        # Short entry conditions
        short_entry = (price < s3) and (price < ema_trend) and vol_ok
        
        if position == 0:
            if long_entry:
                position = 1
                signals[i] = 0.25
                highest_since_entry = price
            elif short_entry:
                position = -1
                signals[i] = -0.25
                lowest_since_entry = price
        elif position == 1:
            highest_since_entry = max(highest_since_entry, price)
            # Trailing stop: exit if price drops below highest - 2.5*ATR
            if price < highest_since_entry - 2.5 * atr_val:
                position = 0
                signals[i] = 0.0
            # Reverse signal on opposite break
            elif short_entry:
                position = -1
                signals[i] = -0.25
                lowest_since_entry = price
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, price)
            # Trailing stop: exit if price rises above lowest + 2.5*ATR
            if price > lowest_since_entry + 2.5 * atr_val:
                position = 0
                signals[i] = 0.0
            # Reverse signal on opposite break
            elif long_entry:
                position = 1
                signals[i] = 0.25
                highest_since_entry = price
    
    return signals