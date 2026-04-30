#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Uses discrete sizing 0.25 to balance return and risk. Target: 50-150 total trades over 4 years (12-37/year).
# 12h timeframe reduces noise and overtrading. Camarilla R3/S3 provides strong support/resistance levels.
# 1w EMA50 filters for primary trend direction to avoid counter-trend whipsaw.
# Volume spike ensures institutional participation. Works in both bull (breakouts with trend) and bear (mean reversion at extremes).

name = "12h_Camarilla_R3S3_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate 12h Camarilla levels (based on prior day's range)
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
    camarilla_r3 = daily_close + 1.1 * hl_range / 4
    camarilla_s3 = daily_close - 1.1 * hl_range / 4
    
    # Calculate 1w EMA(50) for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 24-period average (strict to reduce trades)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 24, 50, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_24[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Camarilla break and 1w EMA50 trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above R3 + close above 1w EMA50
                if curr_close > curr_r3 and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below S3 + close below 1w EMA50
                elif curr_close < curr_s3 and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below S3 OR loses 1w trend
            if curr_low <= stop_loss or curr_close < curr_s3 or curr_close < curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above R3 OR loses 1w trend
            if curr_high >= stop_loss or curr_close > curr_r3 or curr_close > curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals