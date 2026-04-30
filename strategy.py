#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using Williams %R extreme readings on weekly timeframe with volume confirmation and 1w EMA(34) trend filter
# Williams %R identifies overbought/oversold conditions on weekly chart, effective in both bull and bear markets.
# Volume confirmation on daily ensures participation, 1w EMA(34) aligns with longer-term trend to avoid counter-trend trades.
# Designed for low trade frequency (~15-25/year) to minimize fee drag and improve bear market performance.

name = "1d_WeeklyWilliamsR_Extreme_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Williams %R calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w Williams %R (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_1w) / (highest_high - lowest_low)) * -100
    
    # Align Williams %R to 1d timeframe (wait for completed 1w bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w_s = pd.Series(close_1w)
    ema_34_1w = close_1w_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_ema = ema_34_1w_aligned[i]
        curr_atr = atr[i]
        curr_williams_r = williams_r_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and extreme Williams %R with trend alignment
            if volume_spike:
                # Bullish entry: Williams %R oversold (< -80) with 1w uptrend
                if curr_williams_r < -80 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R overbought (> -20) with 1w downtrend
                elif curr_williams_r > -20 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR Williams %R becomes overbought
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams_r > -20:  # exit long on overbought condition
                signals[i] = 0.0
                position = 0
            # Take profit: Williams %R reaches midpoint (-50)
            elif curr_williams_r >= -50:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR Williams %R becomes oversold
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams_r < -80:  # exit short on oversold condition
                signals[i] = 0.0
                position = 0
            # Take profit: Williams %R reaches midpoint (-50)
            elif curr_williams_r <= -50:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals