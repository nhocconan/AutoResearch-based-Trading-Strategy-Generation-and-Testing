#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation + ATR trailing stop.
# Long when price breaks above Donchian upper band AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-period median volume.
# Short when price breaks below Donchian lower band AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-period median volume.
# Exit on ATR trailing stop (2.5x ATR from extreme) or Donchian breakout in opposite direction.
# Donchian channels provide clear breakout levels; 1w EMA50 filters counter-trend trades; volume confirms institutional participation.
# Target: 15-30 trades/year on 1d timeframe. Works in bull (buy breakouts) and bear (sell breakdowns).

name = "1d_Donchian20_Breakout_1wEMA50_VolumeConfirm_ATR_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter (loaded once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 25-period ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian channels from previous day (using daily OHLC)
    # We'll use the previous day's high, low to calculate today's bands
    # Since we're on 1d timeframe, we need to get daily data and align it
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels for each day: highest high and lowest low of last 20 days
    # Donchian Upper = MAX(high of last 20 days)
    # Donchian Lower = MIN(low of last 20 days)
    # We use the previous 20 days to avoid look-ahead
    high_series = pd.Series(df_1d_ohlc['high'])
    low_series = pd.Series(df_1d_ohlc['low'])
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (already aligned, but ensure proper shifting)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d_ohlc, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d_ohlc, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for EMA50, ATR, volume median, and Donchian
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 1w EMA50 direction
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Donchian breakout conditions
        breakout_up = curr_close > donchian_upper_aligned[i]  # break above upper band
        breakout_down = curr_close < donchian_lower_aligned[i]  # break below lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume spike
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Breakout down AND downtrend AND volume spike
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.25
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
            
            # Exit conditions: ATR trailing stop OR Donchian breakout down
            stop_price = highest_since_entry - 2.5 * curr_atr
            if curr_close < stop_price or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry for trailing stop
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR trailing stop OR Donchian breakout up
            stop_price = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stop_price or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals