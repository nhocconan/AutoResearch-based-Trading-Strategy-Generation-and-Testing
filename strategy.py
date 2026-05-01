#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian AND 12h close > EMA50 AND volume > 1.5x 20-bar average.
# Short when price breaks below lower Donchian AND 12h close < EMA50 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Volume spike threshold set to 1.5x to reduce false breakouts and improve signal quality.
# Works in bull markets (trend continuation) and bear markets (mean reversion at extremes).
# Primary timeframe: 4h, HTF: 12h for trend filter.

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 12h EMA50 calculation
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Donchian channels (20-bar lookback on 4h data)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 4h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > high_roll[i]  # break above upper channel
        breakout_down = curr_low < low_roll[i]  # break below lower channel
        
        # Trend filter: use 12h close vs its EMA50 for bias
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        bullish_bias = close_12h_aligned[i] > ema_aligned[i]  # 12h close above its EMA50 = bullish
        bearish_bias = close_12h_aligned[i] < ema_aligned[i]  # 12h close below its EMA50 = bearish
        
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
            if (curr_low < low_roll[i] or 
                bearish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper channel (stoploss) OR bullish bias (trend change)
            if (curr_high > high_roll[i] or 
                bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals