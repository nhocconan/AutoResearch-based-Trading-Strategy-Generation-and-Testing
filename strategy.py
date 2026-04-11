#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_keltner_breakout_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Keltner Channel (20-period EMA, ATR*2)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA(20) for middle band
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(20) for bands
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    upper_1d = ema_20 + 2 * atr_1d
    lower_1d = ema_20 - 2 * atr_1d
    
    # Shift by 1 to use only completed daily bars (previous day's levels)
    upper_1d = np.roll(upper_1d, 1)
    lower_1d = np.roll(lower_1d, 1)
    upper_1d[0] = np.nan
    lower_1d[0] = np.nan
    
    # Align daily Keltner levels to 12h timeframe
    upper_12h = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_12h = align_htf_to_ltf(prices, df_1d, lower_1d)
    ema_12h = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # 12h ATR for volatility filter (14 period)
    tr1_12h = high[1:] - low[1:]
    tr2_12h = np.abs(high[1:] - close[:-1])
    tr3_12h = np.abs(low[1:] - close[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or
            np.isnan(ema_12h[i]) or np.isnan(atr_12h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: price breaks above upper Keltner with volume
        long_signal = volume_confirmed and (price_high > upper_12h[i])
        
        # Short conditions: price breaks below lower Keltner with volume
        short_signal = volume_confirmed and (price_low < lower_12h[i])
        
        # Exit when price returns to the EMA (mean reversion)
        exit_long = position == 1 and price_close < ema_12h[i]
        exit_short = position == -1 and price_close > ema_12h[i]
        
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h Keltner Channel breakout with daily timeframe context.
# Uses daily Keltner Channel (EMA20 ± 2*ATR20) from previous day for longer-term structure.
# Enters long when 12h price breaks above daily upper Keltner with volume >1.5x 20-period average.
# Enters short when 12h price breaks below daily lower Keltner with volume >1.5x 20-period average.
# Exits when price returns to the daily EMA20 (mean reversion within the channel).
# Daily timeframe reduces noise and false breakouts compared to 12h channels alone.
# Volume confirmation filters out low-conviction breakouts.
# Position size: 0.25 to balance risk and return, limiting drawdown in volatile markets.
# Designed to work in both bull and bear markets by adapting to daily volatility ranges.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.