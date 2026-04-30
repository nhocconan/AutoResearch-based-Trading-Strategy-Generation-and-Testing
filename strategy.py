#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extremes with volume confirmation and ADX trend filter
# Williams %R identifies overbought/oversold conditions (>80 oversold, <20 overbought).
# Breakouts from extremes with volume spike indicate strong momentum continuation.
# ADX(14) > 25 ensures we only trade in trending markets to avoid chop.
# Designed for low trade frequency (~25-40/year) to minimize fee drag.
# Works in both bull and bear markets by trading with the trend from extreme levels.

name = "4h_WilliamsR_Extreme_ADXTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high - close_1d) / (highest_high - lowest_low + 1e-10))
    
    # Align Williams %R to 4h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d ADX(14) for trend filter
    plus_dm = np.diff(high_1d)
    minus_dm = np.diff(low_1d)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(high_1d))
    tr2 = np.abs(np.diff(low_1d))
    tr3 = np.abs(np.diff(close_1d))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate ATR(14) for 4h dynamic stoploss
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.max([tr1_4h[0], tr2_4h[0], tr3_4h[0]])], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 30  # warmup for Williams %R and ADX
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_atr = atr_4h[i]
        curr_williams = williams_r_aligned[i]
        curr_adx = adx_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and strong trend (ADX > 25)
            if volume_spike and curr_adx > 25:
                # Bullish entry: Williams %R < -80 (oversold) and price rising
                if curr_williams < -80 and curr_close > close[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R > -20 (overbought) and price falling
                elif curr_williams > -20 and curr_close < close[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR Williams %R > -20 (overbought)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR Williams %R < -80 (oversold)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals