#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation.
    # Works in bull markets (breakouts above H3/H4) and bear markets (breakouts below L3/L4)
    # by following the 4h EMA trend. Uses discrete position size 0.20 to minimize fee churn.
    # Target: 60-150 total trades over 4 years = 15-37/year for 1h.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for Camarilla pivots (based on previous day's range)
    # We'll use 1h timeframe but calculate pivots from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get 4h data for EMA trend filter and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # H4 = close + 1.5*(high - low)
    # H3 = close + 1.125*(high - low)
    # L3 = close - 1.125*(high - low)
    # L4 = close - 1.5*(high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First value invalid
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.125 * (prev_high - prev_low)
    L3 = prev_close - 1.125 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Calculate 4h EMA (21-period) for trend filter
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Calculate 4h volume mean (20-period) with min_periods
    volume_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 1h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    ema_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Align 4h volume for spike detection
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(H4_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
            
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 1.8 * 20-period mean (volume spike)
        volume_confirmation = volume_4h_aligned[i] > 1.8 * vol_ma_aligned[i]
        
        # Trend filter: price above/below 4h EMA indicates trend direction
        trend_up = close[i] > ema_aligned[i]
        trend_down = close[i] < ema_aligned[i]
        
        # Entry conditions: price breaks Camarilla levels with volume confirmation and trend filter
        long_entry = (close[i] > H4_aligned[i] and volume_confirmation and trend_up)
        short_entry = (close[i] < L4_aligned[i] and volume_confirmation and trend_down)
        
        # Exit conditions: price returns to opposite Camarilla level (mean reversion exit)
        long_exit = close[i] < L3_aligned[i]
        short_exit = close[i] > H3_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_1d_4h_camarilla_breakout_volume_ema_v1"
timeframe = "1h"
leverage = 1.0