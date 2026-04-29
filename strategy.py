#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA50 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions: long when %R crosses above -80 from below,
# short when %R crosses below -20 from above. Only takes entries in direction of 1d EMA50 trend.
# Volume confirmation (>2.0x 20-period average) filters weak signals.
# Designed for ~12-30 trades/year on 6h timeframe to minimize fee drag while capturing reversals.
# Works in both bull and bear markets via 1d trend filter - only takes pullbacks in trend direction.

name = "6h_WilliamsR_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate 20-period average volume for confirmation (on 6h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Volume MA and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_williams_r = williams_r[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Williams %R crosses above -20 (overbought)
            if curr_close < entry_price - 2.0 * curr_atr or curr_williams_r > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Williams %R crosses below -80 (oversold)
            if curr_close > entry_price + 2.0 * curr_atr or curr_williams_r < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: Williams %R crosses above -80 from below in uptrend (price > 1d EMA50)
            if vol_confirm and curr_close > curr_ema50_1d:
                if curr_williams_r > -80 and williams_r[i-1] <= -80:  # Cross above -80
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            # Short entry: Williams %R crosses below -20 from above in downtrend (price < 1d EMA50)
            elif vol_confirm and curr_close < curr_ema50_1d:
                if curr_williams_r < -20 and williams_r[i-1] >= -20:  # Cross below -20
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals