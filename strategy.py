#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Uses 1d EMA34 for robust long-term trend alignment (works in both bull and bear markets).
# Long when price breaks above Donchian upper band (20-period high) AND price > 1d EMA34 AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower band (20-period low) AND price < 1d EMA34 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# 1d timeframe provides stable trend filter less prone to whipsaw vs shorter HTF.
# Volume confirmation ensures only high-conviction breakouts are traded.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Donchian20_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d trend: price above/below EMA34
    price_above_ema = close > ema_34_aligned
    price_below_ema = close < ema_34_aligned
    
    # Calculate Donchian channels (20-period) on 12h data
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 12h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Donchian breakout signals
        breakout_up = curr_high > high_ma[i]  # break above upper band
        breakout_down = curr_low < low_ma[i]  # break below lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper band AND price > 1d EMA34 AND volume confirmation
            if (breakout_up and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band AND price < 1d EMA34 AND volume confirmation
            elif (breakout_down and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower band (stoploss) OR price < 1d EMA34 (trend change)
            if (curr_low < low_ma[i] or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (stoploss) OR price > 1d EMA34 (trend change)
            if (curr_high > high_ma[i] or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals