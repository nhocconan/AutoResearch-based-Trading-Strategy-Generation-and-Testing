#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 AND 4h EMA50 uptrend AND volume > 1.5x 20-period median.
# Short when price breaks below S3 AND 4h EMA50 downtrend AND volume > 1.5x 20-period median.
# Camarilla levels provide precise intraday support/resistance; 4h EMA50 filters for higher-timeframe trend; volume confirms breakout conviction.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years) to minimize fee drag.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_v1"
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
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla levels (based on previous day's OHLC)
    # Group by date to get daily OHLC
    df = prices.copy()
    df['date'] = pd.to_datetime(df['open_time']).dt.date
    daily_ohlc = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    # Map daily OHLC to each 1h bar
    df['date_only'] = pd.to_datetime(df['open_time']).dt.date
    daily_map = df.set_index('date_only')[['high', 'low', 'close']].to_dict('index')
    
    # Calculate Camarilla levels for each bar
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        date_key = pd.to_datetime(open_time[i]).date()
        if date_key in daily_map:
            d = daily_map[date_key]
            daily_high = d['high']
            daily_low = d['low']
            daily_close = d['close']
            daily_range = daily_high - daily_low
            
            camarilla_r3[i] = daily_close + daily_range * 1.1 / 4
            camarilla_s3[i] = daily_close - daily_range * 1.1 / 4
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Session filter: 08-20 UTC (reduces noise trades)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Start after warmup for EMA and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 4h EMA50 direction
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND uptrend AND volume spike
            if curr_close > camarilla_r3[i] and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below S3 AND downtrend AND volume spike
            elif curr_close < camarilla_s3[i] and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 OR trend turns down
            if curr_close < camarilla_s3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR trend turns up
            if curr_close > camarilla_r3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals