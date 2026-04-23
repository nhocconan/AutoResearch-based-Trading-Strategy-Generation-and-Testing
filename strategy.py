#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 and close > 4h EMA34 (uptrend) with volume > 1.8x average.
Short when price breaks below Camarilla S3 and close < 4h EMA34 (downtrend) with volume > 1.8x average.
Exit on opposite Camarilla band break or trend reversal. Uses 1h timeframe targeting 60-150 total trades over 4 years.
Camarilla provides intraday support/resistance levels, EMA34 filters medium-term trend, volume spike confirms breakout strength.
Designed to capture momentum moves while avoiding whipsaws in choppy markets across both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1h data for Camarilla calculation (based on previous day's range) - ONCE before loop
    # We need to group by day to get previous day's high/low/close
    prices_df = prices.copy()
    prices_df['date'] = prices_df['open_time'].dt.date
    # Group by date to get daily OHLC
    daily = prices_df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day based on previous day's OHLC
    camarilla_R3 = np.full(len(prices), np.nan)
    camarilla_S3 = np.full(len(prices), np.nan)
    
    for i in range(1, len(daily)):
        prev_high = daily.iloc[i-1]['high']
        prev_low = daily.iloc[i-1]['low']
        prev_close = daily.iloc[i-1]['close']
        
        # Camarilla levels
        R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
        S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
        
        # Apply to all bars of current day
        current_date = daily.iloc[i]['date']
        mask = prices_df['date'] == current_date
        camarilla_R3[mask] = R3
        camarilla_S3[mask] = S3
    
    # Load 4h data for EMA34 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1h timeframe
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume average (24-period) on primary timeframe (1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            hours[i] < 8 or hours[i] > 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        R3_val = camarilla_R3[i]
        S3_val = camarilla_S3[i]
        ema34_val = ema34_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > 4h EMA34 (uptrend) AND volume spike
            if (price > R3_val and price > ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S3 AND price < 4h EMA34 (downtrend) AND volume spike
            elif (price < S3_val and price < ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3 OR trend reversal
                if (price < S3_val or price < ema34_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Camarilla R3 OR trend reversal
                if (price > R3_val or price > ema34_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3_S3_4hEMA34_VolumeSpike"
timeframe = "1h"
leverage = 1.0