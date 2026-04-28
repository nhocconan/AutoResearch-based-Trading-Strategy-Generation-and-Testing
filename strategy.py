#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Enter long when price breaks above R3 with 1d EMA34 uptrend and volume > 1.5x 20-bar average.
# Enter short when price breaks below S3 with 1d EMA34 downtrend and volume > 1.5x 20-bar average.
# Exit when price retreats to the Camarilla pivot point (PP).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 100-200 total trades over 4 years (25-50/year).
# Camarilla levels provide intraday support/resistance; 1d EMA34 ensures higher timeframe alignment;
# volume confirmation filters weak breakouts. Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) for dynamic volume threshold (optional)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day
    # For each 4h bar, use the prior day's high, low, close
    # We'll calculate daily OHLC first
    df = prices.copy()
    df['date'] = df['open_time'].dt.date
    daily_agg = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily_agg) < 2:
        return np.zeros(n)
    
    # Map daily data to each 4h bar
    high_pd = []
    low_pd = []
    close_pd = []
    
    for dt in df['open_time']:
        date_key = dt.date()
        day_row = daily_agg[daily_agg['date'] == date_key]
        if len(day_row) > 0:
            high_pd.append(day_row.iloc[0]['high'])
            low_pd.append(day_row.iloc[0]['low'])
            close_pd.append(day_row.iloc[0]['close'])
        else:
            # Find previous day
            prev_dates = daily_agg[daily_agg['date'] < date_key]
            if len(prev_dates) > 0:
                prev_row = prev_dates.iloc[-1]
                high_pd.append(prev_row['high'])
                low_pd.append(prev_row['low'])
                close_pd.append(prev_row['close'])
            else:
                high_pd.append(high[0])  # fallback
                low_pd.append(low[0])
                close_pd.append(close[0])
    
    high_pd = np.array(high_pd)
    low_pd = np.array(low_pd)
    close_pd = np.array(close_pd)
    
    # Calculate Camarilla levels
    # PP = (H + L + C) / 3
    pp = (high_pd + low_pd + close_pd) / 3
    # R3 = C + (H - L) * 1.1 / 2
    r3 = close_pd + (high_pd - low_pd) * 1.1 / 2
    # S3 = C - (H - L) * 1.1 / 2
    s3 = close_pd - (high_pd - low_pd) * 1.1 / 2
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(pp[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > R3, EMA34 up, volume confirm
            if price > r3[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < S3, EMA34 down, volume confirm
            elif price < s3[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at PP
            if price <= pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at PP
            if price >= pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals