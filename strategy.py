#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel AND 1d close > EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below lower Donchian channel AND 1d close < EMA50 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Volume spike threshold set to 2.0x to reduce false breakouts and improve signal quality.
# Works in bull markets (trend continuation) and bear markets (mean reversion at extremes).
# Primary timeframe: 12h, HTF: 1d for trend filter.

name = "12h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 1d close aligned for trend bias
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    # We need to calculate this on the primary timeframe (12h) using rolling window
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 12h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_aligned[i]) or np.isnan(close_1d_aligned[i]) or \
           np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > upper_channel[i]  # break above upper channel
        breakout_down = curr_low < lower_channel[i]  # break below lower channel
        
        # Trend filter: use 1d close vs its EMA50 for bias
        bullish_bias = close_1d_aligned[i] > ema_aligned[i]  # 1d close above its EMA50 = bullish
        bearish_bias = close_1d_aligned[i] < ema_aligned[i]  # 1d close below its EMA50 = bearish
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper channel AND bullish bias AND volume confirmation
            if (breakout_up and 
                bullish_bias and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower channel AND bearish bias AND volume confirmation
            elif (breakout_down and 
                  bearish_bias and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower channel (stoploss) OR bearish bias (trend change)
            if (curr_low < lower_channel[i] or 
                bearish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper channel (stoploss) OR bullish bias (trend change)
            if (curr_high > upper_channel[i] or 
                bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals