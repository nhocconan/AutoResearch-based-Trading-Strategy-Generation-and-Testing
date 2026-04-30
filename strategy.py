#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses discrete sizing 0.20 to minimize fee drag. Target: 60-150 total trades over 4 years (15-37/year).
# Long when price breaks above Camarilla R3 AND price > 4h EMA50 AND volume spike.
# Short when price breaks below Camarilla S3 AND price < 4h EMA50 AND volume spike.
# ATR-based stoploss: exit when price moves against position by 2.0 * ATR(14).
# Session filter (08-20 UTC) to reduce noise. Works in bull via breakout longs, in bear via breakdown shorts.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike_ATRStop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # Typical Camarilla uses previous day's high, low, close
    # For intraday, we approximate using rolling window of 24 periods (1d at 1h)
    roll_high_24 = pd.Series(high).rolling(window=24, min_periods=24).max().shift(1)  # previous day high
    roll_low_24 = pd.Series(low).rolling(window=24, min_periods=24).min().shift(1)   # previous day low
    roll_close_24 = pd.Series(close).rolling(window=24, min_periods=24).mean().shift(1)  # previous day close approx
    
    # Avoid NaN in first bars
    roll_high_24 = np.where(np.isnan(roll_high_24), high, roll_high_24)
    roll_low_24 = np.where(np.isnan(roll_low_24), low, roll_low_24)
    roll_close_24 = np.where(np.isnan(roll_close_24), close, roll_close_24)
    
    # Camarilla R3 and S3
    camarilla_range = roll_high_24 - roll_low_24
    camarilla_r3 = roll_close_24 + camarilla_range * 1.1 / 4
    camarilla_s3 = roll_close_24 - camarilla_range * 1.1 / 4
    
    # Calculate 4h EMA(50) for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 2.0x 24-period average (1 day at 1h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(24, 24, 50, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_24[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_camarilla_r3 = camarilla_r3[i]
        curr_camarilla_s3 = camarilla_s3[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above Camarilla R3 AND above 4h EMA50
                if (curr_close > curr_camarilla_r3 and 
                    curr_close > curr_ema_50_4h):
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Camarilla S3 AND below 4h EMA50
                elif (curr_close < curr_camarilla_s3 and 
                      curr_close < curr_ema_50_4h):
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # ATR-based stoploss: exit when price drops below entry - 2.0 * ATR
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # ATR-based stoploss: exit when price rises above entry + 2.0 * ATR
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals