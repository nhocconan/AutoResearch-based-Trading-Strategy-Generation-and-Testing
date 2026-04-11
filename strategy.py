#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h ATR for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Load daily data for previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.166
    s3_1d = close_1d - range_1d * 1.166
    r3_1d = np.roll(r3_1d, 1)
    s3_1d = np.roll(s3_1d, 1)
    r3_1d[0] = np.nan
    s3_1d[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Daily volume ratio filter
    vol_1d = df_1d['volume'].values
    vol_ma_10_1d = pd.Series(vol_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_10_1d)
    
    # 4h volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or
            np.isnan(atr_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(vol_ma_10_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        vol_daily_ma = vol_ma_10_1d_aligned[i]
        
        # Volume confirmation
        volume_confirmed = (volume_current > 1.5 * vol_ma) and (volume_current > 2.0 * vol_daily_ma)
        
        # Long: break above R3 with volume and price above 12h ATR mean (trend filter)
        long_signal = volume_confirmed and (price_high > r3_4h[i]) and (price_close > np.nanmean(atr_12h_aligned[max(0, i-100):i+1]))
        
        # Short: break below S3 with volume and price below 12h ATR mean
        short_signal = volume_confirmed and (price_low < s3_4h[i]) and (price_close < np.nanmean(atr_12h_aligned[max(0, i-100):i+1]))
        
        # Exit at opposite level
        exit_long = position == 1 and price_close < s3_4h[i]
        exit_short = position == -1 and price_close > r3_4h[i]
        
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

# Hypothesis: 4h Camarilla breakout using previous day's R3/S3 levels with dual volume confirmation (4h and daily) and 12h ATR trend filter.
# Enters long when 4h price breaks above daily R3 with volume >1.5x 4h 20-period average AND >2x daily 10-period average
# and price above recent 12h ATR mean (indicating bullish momentum). Enters short when 4h price breaks below daily S3
# with same volume conditions and price below recent 12h ATR mean. Exits when price reaches the opposite level.
# Uses 12h ATR as a trend filter to avoid counter-trend trades in ranging markets.
# Dual volume filter reduces false breakouts, targeting 20-40 trades per year per symbol.
# Position size: 0.25 to balance risk and return, limiting drawdown in volatile markets.
# Designed to work in both bull and bear markets by combining intraday breakouts with higher timeframe trend filter.
# Target: 50-100 trades over 4 years (12-25/year) to minimize fee drag.