#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter (EMA34) and volume confirmation
# Long when price breaks above Camarilla R3 + 4h EMA34 uptrend + volume > 1.5x 20-period avg
# Short when price breaks below Camarilla S3 + 4h EMA34 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.20) to balance return and drawdown control.
# Camarilla pivots provide intraday support/resistance levels that work in ranging markets.
# 4h EMA34 filters trend direction to avoid counter-trend trades.
# Volume confirmation ensures breakouts have participation.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    # === 4h Indicator: EMA34 for trend filter ===
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1h Camarilla Pivot Points (using previous day's OHLC) ===
    # Calculate daily OHLC from 1h data
    # Group by date to get daily high, low, close
    df = prices.copy()
    df['date'] = df['open_time'].dt.date
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.5 / 2)
    # R3 = close + ((high - low) * 1.25 / 2)
    # R2 = close + ((high - low) * 1.1 / 2)
    # R1 = close + ((high - low) * 0.5 / 2)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 0.5 / 2)
    # S2 = close - ((high - low) * 1.1 / 2)
    # S3 = close - ((high - low) * 1.25 / 2)
    # S4 = close - ((high - low) * 1.5 / 2)
    
    # We'll use R3 and S3 for breakouts
    daily['range'] = daily['high'] - daily['low']
    daily['R3'] = daily['close'] + (daily['range'] * 1.25 / 2)
    daily['S3'] = daily['close'] - (daily['range'] * 1.25 / 2)
    
    # Map daily levels to each 1h bar
    # Create arrays of same length as prices
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    
    # For each day, fill the corresponding hours
    for idx, row in daily.iterrows():
        day = row['date']
        # Get indices for this day
        day_mask = df['date'] == day
        if day_mask.any():
            camarilla_R3[day_mask] = row['R3']
            camarilla_S3[day_mask] = row['S3']
    
    # Forward fill to handle any missing values (first bar of day)
    camarilla_R3 = pd.Series(camarilla_R3).ffill().bfill().values
    camarilla_S3 = pd.Series(camarilla_S3).ffill().bfill().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20) + 2  # EMA(34) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (close > R3)
        # 2. 4h EMA34 uptrend (price > EMA)
        # 3. Volume confirmation
        if (close[i] > camarilla_R3[i]) and \
           (close[i] > ema_4h_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (close < S3)
        # 2. 4h EMA34 downtrend (price < EMA)
        # 3. Volume confirmation
        elif (close[i] < camarilla_S3[i]) and \
             (close[i] < ema_4h_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R3S3_4hEMA34_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0