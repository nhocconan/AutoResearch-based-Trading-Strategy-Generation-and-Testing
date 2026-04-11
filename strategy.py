#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_keltner_breakout_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Calculate 12h Keltner Channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Keltner Channel: EMA(20) ± ATR(10) * 2
    ema_20 = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr = np.maximum(high_12h - low_12h,
                    np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                               np.abs(low_12h - np.roll(close_12h, 1))))
    tr[0] = high_12h[0] - low_12h[0]
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    kc_upper = ema_20 + (atr_10 * 2)
    kc_lower = ema_20 - (atr_10 * 2)
    
    # Shift by 1 to use only completed 12h bars
    ema_20 = np.roll(ema_20, 1)
    kc_upper = np.roll(kc_upper, 1)
    kc_lower = np.roll(kc_lower, 1)
    ema_20[0] = np.nan
    kc_upper[0] = np.nan
    kc_lower[0] = np.nan
    
    # Align 12h Keltner to 4h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    kc_upper_aligned = align_htf_to_ltf(prices, df_12h, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_12h, kc_lower)
    
    # Volume confirmation: volume > 1.5x 20-period average on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_aligned[i]) or np.isnan(kc_upper_aligned[i]) or
            np.isnan(kc_lower_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend direction from 12h EMA
        uptrend = price_close > ema_20_aligned[i]
        downtrend = price_close < ema_20_aligned[i]
        
        # Long: price breaks above upper Keltner with volume in uptrend
        long_signal = volume_confirmed and uptrend and (price_high > kc_upper_aligned[i])
        
        # Short: price breaks below lower Keltner with volume in downtrend
        short_signal = volume_confirmed and downtrend and (price_low < kc_lower_aligned[i])
        
        # Exit when price returns to 12h EMA (mean reversion)
        exit_long = position == 1 and price_close < ema_20_aligned[i]
        exit_short = position == -1 and price_close > ema_20_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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

# Hypothesis: Keltner breakout strategy on 4h timeframe using 12h Keltner Channels.
# Enters long when price breaks above upper Keltner (EMA20 + 2*ATR10) with volume confirmation (>1.5x avg volume) during uptrend (price > 12h EMA20).
# Enters short when price breaks below lower Keltner (EMA20 - 2*ATR10) with volume confirmation during downtrend (price < 12h EMA20).
# Exits when price returns to the 12h EMA20 (mean reversion to the trend).
# Works in both bull and bear markets by trading breakouts in the direction of the 12h trend.
# Volume confirmation reduces false breakouts. Trend filter ensures we trade with higher timeframe momentum.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.