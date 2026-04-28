#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla levels provide high-probability reversal/breakout zones. R3/S3 are strong
# support/resistance levels. Price breaking above R3 with bullish 1d EMA34 trend and
# volume spike indicates strong momentum continuation. Price breaking below S3 with
# bearish 1d EMA34 trend and volume spike indicates strong momentum continuation down.
# Works in both bull and bear markets by following the higher-timeframe trend while
# using precise Camarilla breakouts for entry. Target 12-37 trades/year to minimize fee drag.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Get daily data for Camarilla pivot calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    #            S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    df_1d = df_1d.copy()
    df_1d['prior_high'] = df_1d['high'].shift(1)
    df_1d['prior_low'] = df_1d['low'].shift(1)
    df_1d['prior_close'] = df_1d['close'].shift(1)
    
    df_1d['cam_R3'] = df_1d['prior_close'] + ((df_1d['prior_high'] - df_1d['prior_low']) * 1.1 / 4)
    df_1d['cam_S3'] = df_1d['prior_close'] - ((df_1d['prior_high'] - df_1d['prior_low']) * 1.1 / 4)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA34 to 12h timeframe (completed daily values only)
    cam_R3_aligned = align_htf_to_ltf(prices, df_1d, df_1d['cam_R3'].values)
    cam_S3_aligned = align_htf_to_ltf(prices, df_1d, df_1d['cam_S3'].values)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cam_R3_aligned[i]) or np.isnan(cam_S3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        r3 = cam_R3_aligned[i]
        s3 = cam_S3_aligned[i]
        ema_trend = ema_34_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > EMA34 (bullish trend) AND volume spike
            if price > r3 and price > ema_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Camarilla S3 AND price < EMA34 (bearish trend) AND volume spike
            elif price < s3 and price < ema_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or breakdown below S3
            # ATR-based stoploss: 2.0 * ATR below entry (using 12h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or price breaks below Camarilla S3 (trend reversal)
            if price < stop_loss or price < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or breakout above R3
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or price breaks above Camarilla R3 (trend reversal)
            if price > stop_loss or price > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals