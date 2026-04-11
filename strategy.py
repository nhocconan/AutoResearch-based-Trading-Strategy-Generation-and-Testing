#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 14-period ATR on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Keltner Channels on 4h data (20-period EMA, 2.0 ATR multiplier)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_20_4h = pd.Series(high - low).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_20 + 2.0 * atr_20_4h
    keltner_lower = ema_20 - 2.0 * atr_20_4h
    
    # Volume confirmation: 20-period average
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Higher timeframe volatility filter: only trade when 1d ATR is elevated
        # (indicates active market, avoids choppy sideways periods)
        if i >= 20:  # Need enough history for ATR average
            atr_ma_20 = np.nanmean(atr_14_1d_aligned[max(0, i-19):i+1])
            vol_filter = atr_14_1d_aligned[i] > 0.8 * atr_ma_20
        else:
            vol_filter = True
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Keltner upper + volume confirmation + volatility filter
        if price_close > keltner_upper[i] and vol_confirm and vol_filter:
            enter_long = True
        
        # Short: Price breaks below Keltner lower + volume confirmation + volatility filter
        if price_close < keltner_lower[i] and vol_confirm and vol_filter:
            enter_short = True
        
        # Exit conditions: price crosses back through the EMA (middle line)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price crosses below EMA20
            exit_long = price_close < ema_20[i]
        elif position == -1:
            # Exit short if price crosses above EMA20
            exit_short = price_close > ema_20[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals