#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and volume confirmation.
# Long when price breaks above Camarilla R1 AND 4h close > EMA20 (bullish trend) AND volume > 1.8x 20-bar average.
# Short when price breaks below Camarilla S1 AND 4h close < EMA20 (bearish trend) AND volume > 1.8x 20-bar average.
# Uses discrete sizing 0.20 to manage drawdown. Target: 80-140 total trades over 4 years (20-35/year).
# Primary timeframe: 1h, HTF: 4h for EMA trend and Camarilla calculation.
# Session filter: 08-20 UTC to reduce noise trades.

name = "1h_Camarilla_R1S1_Breakout_4hEMA20_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Camarilla levels and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 4h bar's OHLC
    prev_close_4h = df_4h['close'].values
    prev_high_4h = df_4h['high'].values
    prev_low_4h = df_4h['low'].values
    
    # Camarilla levels: R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12)
    camarilla_r1_4h = prev_close_4h + ((prev_high_4h - prev_low_4h) * 1.1 / 12)
    camarilla_s1_4h = prev_close_4h - ((prev_high_4h - prev_low_4h) * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # 4h EMA20 trend filter
    ema_20_4h = pd.Series(prev_close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: current 1h volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (precompute hour array)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)  # Volume spike threshold
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r1_aligned[i]  # break above R1
        breakout_down = curr_low < camarilla_s1_aligned[i]  # break below S1
        
        # Trend filter: bullish if close > EMA20, bearish if close < EMA20
        bullish_trend = curr_close > ema_20_aligned[i]
        bearish_trend = curr_close < ema_20_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R1 AND bullish trend AND volume confirmation
            if (breakout_up and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: breakout below S1 AND bearish trend AND volume confirmation
            elif (breakout_down and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S1 (stoploss) OR trend turns bearish
            if (curr_low < camarilla_s1_aligned[i] or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above R1 (stoploss) OR trend turns bullish
            if (curr_high > camarilla_r1_aligned[i] or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals