#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation
# Long: Close > Donchian Upper(20) AND price > 1w EMA34 AND volume > 2.0x 20-bar avg
# Short: Close < Donchian Lower(20) AND price < 1w EMA34 AND volume > 2.0x 20-bar avg
# Exit: Close crosses Donchian midpoint OR price crosses 1w EMA34 OR ATR stoploss (2.0 * ATR)
# Using 1w HTF for trend filter provides more stable signals than shorter HTF, reducing whipsaws in choppy markets
# Volume spike confirms institutional participation and reduces false breakouts
# Discrete position sizing: 0.30 for long/short, 0.0 for flat to minimize fee churn
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe

name = "1d_Donchian_Breakout_1wEMA34_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 34)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Calculate Donchian channels (based on previous 20 bars)
        if i >= 20:
            # Donchian levels for current bar using previous 20 bars' high/low
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
            donchian_mid = (donchian_high + donchian_low) / 2.0
        else:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: Close below Donchian midpoint OR price below 1w EMA34 OR stoploss hit
            if curr_close < donchian_mid or curr_close < curr_ema_1w or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above Donchian midpoint OR price above 1w EMA34 OR stoploss hit
            if curr_close > donchian_mid or curr_close > curr_ema_1w or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: Close > Donchian Upper AND price > 1w EMA34 AND volume spike
            if (curr_close > donchian_high and 
                curr_close > curr_ema_1w and
                vol_spike):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short entry: Close < Donchian Lower AND price < 1w EMA34 AND volume spike
            elif (curr_close < donchian_low and 
                  curr_close < curr_ema_1w and
                  vol_spike):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals