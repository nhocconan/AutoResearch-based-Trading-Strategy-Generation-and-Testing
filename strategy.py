#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w EMA50 trend filter and volume spike confirmation
# Long when Williams %R < -80 AND price > 1w EMA50 AND volume > 2.0x 20-bar avg
# Short when Williams %R > -20 AND price < 1w EMA50 AND volume > 2.0x 20-bar avg
# Exit on opposite Williams %R level (-20 for longs, -80 for shorts) OR ATR-based stoploss (2.5x ATR)
# Williams %R identifies overbought/oversold conditions. 1w EMA50 filters counter-trend moves.
# Volume spike confirms institutional participation. ATR stoploss manages risk in volatile markets.
# Target: 20-50 trades/year on 1d (80-200 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1w trend while using Williams %R for precise entries.

name = "1d_WilliamsR_1wEMA50_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # EMA50, volume MA, ATR all need warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_1w_aligned[i]
        wr = williams_r[i]
        atr_val = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Check stoploss: close < entry_price - 2.5 * ATR
            if curr_close < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Check exit: Williams %R > -20 (overbought)
            elif wr > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Check stoploss: close > entry_price + 2.5 * ATR
            if curr_close > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Check exit: Williams %R < -80 (oversold)
            elif wr < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R < -80 AND price > 1w EMA50 AND volume confirmation
            if wr < -80 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when Williams %R > -20 AND price < 1w EMA50 AND volume confirmation
            elif wr > -20 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals