#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout + 4h EMA50 trend filter + volume confirmation + ATR trailing stop.
# Long when price breaks above Camarilla R3 AND price > 4h EMA50 (uptrend) AND volume > 1.5x 20-period median.
# Short when price breaks below Camarilla S3 AND price < 4h EMA50 (downtrend) AND volume > 1.5x 20-period median.
# Exit on ATR trailing stop (2.0x ATR from extreme) or Camarilla breakout in opposite direction.
# Camarilla levels provide high-probability reversal/breakout points; 4h EMA50 filters counter-trend trades; volume confirms institutional participation.
# Session filter (08-20 UTC) reduces noise trades outside active market hours.
# Target: 15-37 trades/year on 1h timeframe (~60-150 total over 4 years).
# Works in bull (buy breakouts) and bear (sell breakdowns) by following higher timeframe trend.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeConfirm_ATR_v1"
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
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter (loaded once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 20-period ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels from previous day (using daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day: H, L, C from previous day
    # Camarilla R3 = C + (H-L)*1.1/2
    # Camarilla S3 = C - (H-L)*1.1/2
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for EMA50, ATR, and volume median
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 4h EMA50 direction
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Camarilla breakout conditions
        breakout_up = curr_close > camarilla_r3_aligned[i]  # break above R3
        breakout_down = curr_close < camarilla_s3_aligned[i]  # break below S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume spike
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Breakout down AND downtrend AND volume spike
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry for trailing stop
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR trailing stop OR Camarilla breakout down
            stop_price = highest_since_entry - 2.0 * curr_atr
            if curr_close < stop_price or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Update lowest low since entry for trailing stop
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR trailing stop OR Camarilla breakout up
            stop_price = lowest_since_entry + 2.0 * curr_atr
            if curr_close > stop_price or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals