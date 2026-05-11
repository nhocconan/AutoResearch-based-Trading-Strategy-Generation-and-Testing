#!/usr/bin/env python3
name = "1d_1W_RangeBreakout_WithTrendAndVolume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe trend and range
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high/low for range identification
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Previous week's values (avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = weekly_high[0]
    prev_weekly_low[0] = weekly_low[0]
    prev_weekly_close[0] = weekly_close[0]
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily volatility filter (ATR-based)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_weekly_high[i]) or np.isnan(prev_weekly_low[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr[i] > (np.mean(atr[max(0, i-50):i+1]) * 0.5)
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above previous week's high with volume and weekly uptrend
            if (close[i] > prev_weekly_high[i] and 
                volume_surge and 
                vol_filter and
                close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below previous week's low with volume and weekly downtrend
            elif (close[i] < prev_weekly_low[i] and 
                  volume_surge and 
                  vol_filter and
                  close[i] < ema_21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to weekly range or trend fails
            if position == 1:
                # Exit long: price returns to weekly range or weekly trend turns bearish
                if (close[i] < prev_weekly_high[i]) or (close[i] < ema_21_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to weekly range or weekly trend turns bullish
                if (close[i] > prev_weekly_low[i]) or (close[i] > ema_21_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals