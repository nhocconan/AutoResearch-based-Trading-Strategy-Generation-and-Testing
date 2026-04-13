#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and 1w trend filter
    # Long when price breaks above H3 (1d) + 1d volume > 2x average + price > 1w EMA200
    # Short when price breaks below L3 (1d) + 1d volume > 2x average + price < 1w EMA200
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 75-200 total trades over 4 years (~19-50/year)
    # Camarilla levels provide institutional support/resistance; volume confirms breakout strength
    # Weekly EMA200 filter avoids counter-trend trades in bear markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla levels and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = C + 1.1*(H-L)*1.1/2, H3 = C + 1.1*(H-L)*1.1/4, etc.
    # We'll use H3/L3 for breakout signals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Calculate Camarilla H3 and L3
    # H3 = C + 1.1*(H-L)*1.1/4
    # L3 = C - 1.1*(H-L)*1.1/4
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    # Align 1d Camarilla levels to 4h (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 1w data for EMA200 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2 * 20-period average
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        volume_confirm = vol_1d_aligned[i] > 2.0 * vol_ma_aligned[i]
        
        # Trend filter: price vs 1w EMA200
        price_above_ema = close[i] > ema_200_aligned[i]
        price_below_ema = close[i] < ema_200_aligned[i]
        
        # Entry signals: Camarilla H3/L3 breakout with volume and trend confirmation
        long_entry = (close[i] > camarilla_h3_aligned[i]) and volume_confirm and price_above_ema
        short_entry = (close[i] < camarilla_l3_aligned[i]) and volume_confirm and price_below_ema
        
        # Exit conditions: price returns to Camarilla pivot level (close to previous day's close)
        # Using 1d close as mean reversion target
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
        long_exit = close[i] < prev_close_aligned[i]
        short_exit = close[i] > prev_close_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0