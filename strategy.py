#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA34 trend filter and volume spike
# Uses Williams %R(14) to identify oversold/overbought conditions
# Trend filter uses 1d EMA34 to ensure trades align with higher timeframe momentum
# Volume confirmation (>2.0x 24-period average) ensures institutional participation
# Designed for 4h timeframe to capture swings with controlled trade frequency (~15-35 trades/year)
# BTC/ETH focus: requires both Williams %R extreme and EMA alignment to avoid whipsaws

name = "4h_WilliamsR_MeanRev_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 24-period average volume for confirmation
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(34, 24, 14)  # EMA34, volume MA, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_williams_r = williams_r[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_24[i]
        
        # Handle stoploss and exits
        if position == 1:  # Long position
            # Stoploss: price closes below entry - 2.0 * ATR_at_entry
            if curr_close < entry_price - 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: Williams %R exits oversold territory or trend turns down
            elif curr_williams_r > -20 or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.0 * ATR_at_entry
            if curr_close > entry_price + 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: Williams %R exits overbought territory or trend turns up
            elif curr_williams_r < -80 or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 24-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: Williams %R oversold (< -80) in uptrend (close > EMA34_1d)
            if vol_confirm and curr_close > curr_ema34_1d:
                if curr_williams_r < -80:  # Oversold condition
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry: Williams %R overbought (> -20) in downtrend (close < EMA34_1d)
            elif vol_confirm and curr_close < curr_ema34_1d:
                if curr_williams_r > -20:  # Overbought condition
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals