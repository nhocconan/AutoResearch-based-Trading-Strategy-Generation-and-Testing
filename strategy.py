#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R (14) with 1d EMA34 trend filter and volume spike confirmation
# Williams %R measures overbought/oversold levels: -20 to 0 = overbought, -80 to -100 = oversold
# In bull markets (price > 1d EMA34), we look for Williams %R crossing above -50 from below with volume confirmation for longs
# In bear markets (price < 1d EMA34), we look for Williams %R crossing below -50 from above with volume confirmation for shorts
# Uses volume confirmation (>1.8x 20-period average) to reduce false signals
# Designed for ~25-50 trades/year on 4h timeframe to minimize fee drag while capturing momentum swings
# Works in both bull and bear via 1d EMA34 trend filter - only trades in direction of higher timeframe momentum

name = "4h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 20  # volume MA and Williams %R warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_price[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_williams_r = williams_r[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Previous Williams %R for crossover detection
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Williams %R crosses below -50 (loss of bullish momentum)
            if curr_close < entry_price - 2.0 * curr_atr or curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Williams %R crosses above -50 (loss of bearish momentum)
            if curr_close > entry_price + 2.0 * curr_atr or curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long entry when price > 1d EMA34 (bullish regime) AND Williams %R crosses above -50 from below with volume confirmation
            if curr_close > curr_ema34_1d and prev_williams_r < -50 <= curr_williams_r and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry when price < 1d EMA34 (bearish regime) AND Williams %R crosses below -50 from above with volume confirmation
            elif curr_close < curr_ema34_1d and prev_williams_r > -50 >= curr_williams_r and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals