#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R mean reversion with 4h EMA50 trend filter and volume spike confirmation
# Uses discrete sizing 0.20 to minimize fee drag. Target: 60-150 total trades over 4 years (15-37/year).
# Williams %R identifies overbought/oversold conditions for mean reversion in ranging markets.
# 4h EMA50 filters for higher timeframe trend alignment to avoid counter-trend trades.
# Volume spike ensures institutional participation. Session filter (08-20 UTC) reduces noise.
# Only takes mean reversion trades when price is near prior day's Camarilla S3/R3 levels for structure.

name = "1h_WilliamsR_ME_4hEMA50_VolumeSpike_v1"
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
    
    # Calculate 1h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate 4h EMA(50) for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 2.0x 24-period average (strict to reduce trades)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    # Calculate prior day's Camarilla levels for structure reference
    df = prices.copy()
    df['date'] = pd.DatetimeIndex(open_time).date
    daily_agg = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    date_map = {date: i for i, date in enumerate(daily_agg['date'])}
    daily_high = np.array([daily_agg.loc[daily_agg['date'] == date, 'high'].values[0] 
                          if date in date_map else np.nan 
                          for date in pd.DatetimeIndex(open_time).date])
    daily_low = np.array([daily_agg.loc[daily_agg['date'] == date, 'low'].values[0] 
                         if date in date_map else np.nan 
                         for date in pd.DatetimeIndex(open_time).date])
    daily_close = np.array([daily_agg.loc[daily_agg['date'] == date, 'close'].values[0] 
                           if date in date_map else np.nan 
                           for date in pd.DatetimeIndex(open_time).date])
    
    hl_range = daily_high - daily_low
    camarilla_s3 = daily_close - 1.1 * hl_range / 4  # Support level
    camarilla_r3 = daily_close + 1.1 * hl_range / 4  # Resistance level
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 24, 50, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma_24[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_r3[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_camarilla_s3 = camarilla_s3[i]
        curr_camarilla_r3 = camarilla_r3[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Williams %R extreme and Camarilla proximity
            if curr_volume_spike:
                # Bullish mean reversion: Williams %R oversold (< -80) + near S3 + above 4h EMA50
                if (curr_williams_r < -80 and 
                    curr_close >= curr_camarilla_s3 * 0.995 and  # Near S3 (within 0.5%)
                    curr_close > curr_ema_50_4h):
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish mean reversion: Williams %R overbought (> -20) + near R3 + below 4h EMA50
                elif (curr_williams_r > -20 and 
                      curr_close <= curr_camarilla_r3 * 1.005 and  # Near R3 (within 0.5%)
                      curr_close < curr_ema_50_4h):
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to neutral OR loses 4h trend OR hits Camarilla R3
            if (curr_williams_r > -50 or  # Returned to neutral territory
                curr_close < curr_ema_50_4h or  # Lost 4h uptrend
                curr_close >= curr_camarilla_r3 * 0.995):  # Reached R3 level
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral OR loses 4h trend OR hits Camarilla S3
            if (curr_williams_r < -50 or  # Returned to neutral territory
                curr_close > curr_ema_50_4h or  # Lost 4h downtrend
                curr_close <= curr_camarilla_s3 * 1.005):  # Reached S3 level
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals