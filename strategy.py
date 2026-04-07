#!/usr/bin/env python3
"""
4h_cci_breakout_1d_trend_volume_v1
Hypothesis: On 4-hour timeframe, use CCI (14) for momentum signals, filtered by daily trend (EMA20 > EMA50) and volume confirmation.
Enter long when CCI crosses above +100 and daily trend is up with volume > 1.5x average.
Enter short when CCI crosses below -100 and daily trend is down with volume > 1.5x average.
Exit when CCI returns to neutral zone (-100 to 100) or trend reverses.
Designed to work in both bull and bear markets by following daily trend direction.
Targets 20-50 trades/year to minimize fee drift while capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # CCI (14) calculation
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    ma_tp = tp_series.rolling(window=14, min_periods=14).mean()
    mad = tp_series.rolling(window=14, min_periods=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp_series - ma_tp) / (0.015 * mad)
    cci_values = cci.values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA20 and EMA50 on daily close
    daily_close = df_1d['close'].values
    daily_close_s = pd.Series(daily_close)
    ema20_1d = daily_close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_1d = daily_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align to 4h timeframe (shifted by 1 day to avoid look-ahead)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after CCI warmup
        # Skip if required data not available
        if (np.isnan(cci_values[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from daily: up if EMA20 > EMA50, down if EMA20 < EMA50
        trend_up = ema20_1d_aligned[i] > ema50_1d_aligned[i]
        trend_down = ema20_1d_aligned[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when CCI returns to neutral zone (< 100)
            if cci_values[i] < 100:
                exit_long = True
            # Exit on trend reversal
            elif not trend_up:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when CCI returns to neutral zone (> -100)
            if cci_values[i] > -100:
                exit_short = True
            # Exit on trend reversal
            elif not trend_down:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI crosses above +100, daily trend up, volume confirmation
            long_entry = (cci_values[i] > 100) and (cci_values[i-1] <= 100) and trend_up and vol_confirm
            
            # Short entry: CCI crosses below -100, daily trend down, volume confirmation
            short_entry = (cci_values[i] < -100) and (cci_values[i-1] >= -100) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals