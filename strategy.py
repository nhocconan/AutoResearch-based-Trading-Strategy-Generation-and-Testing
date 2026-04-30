#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R extreme readings with 1d EMA trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; readings below -80 or above -20 with reversal
# capture mean-reversion opportunities in ranging markets while EMA filter ensures alignment with
# longer-term trend to avoid counter-trend trades. Designed for low trade frequency (<30/year) to
# minimize fee drag in both bull and bear markets. Uses 12h timeframe with 1d HTF for Williams %R
# and trend filter.

name = "12h_WilliamsR_Extreme_1dTrend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for Williams %R
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50  # Neutral value when range is zero
    )
    
    # Align Williams %R to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(50) and Williams %R
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 30-period average
        vol_ma_30 = np.mean(volume[max(0, i-30):i])
        volume_spike = volume[i] > (1.8 * vol_ma_30)
        
        curr_close = close[i]
        curr_ema = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_williams_r = williams_r_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and Williams %R extreme
            if volume_spike:
                # Bullish entry: Williams %R crosses above -80 from below (oversold bounce)
                # with 1d uptrend (price above EMA)
                if (curr_williams_r > -80 and 
                    i > start_idx and 
                    williams_r_aligned[i-1] <= -80 and
                    curr_close > curr_ema):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R crosses below -20 from above (overbought rejection)
                # with 1d downtrend (price below EMA)
                elif (curr_williams_r < -20 and 
                      i > start_idx and 
                      williams_r_aligned[i-1] >= -20 and
                      curr_close < curr_ema):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: Williams %R reaches overbought territory (-20) or price crosses below EMA
            elif curr_williams_r >= -20 or curr_close < curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: Williams %R reaches oversold territory (-80) or price crosses above EMA
            elif curr_williams_r <= -80 or curr_close > curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals