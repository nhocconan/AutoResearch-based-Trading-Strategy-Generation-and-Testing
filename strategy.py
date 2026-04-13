#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout + 12h EMA trend + volume spike + ATR stoploss
    # Long: price > Donchian(20) high AND price > 12h EMA50 AND volume > 1.5x 20-period avg
    # Short: price < Donchian(20) low AND price < 12h EMA50 AND volume > 1.5x 20-period avg
    # Exit: ATR-based trailing stop (signal=0 when price < highest_high - 2*ATR for longs,
    #       or price > lowest_low + 2*ATR for shorts)
    # Using 4h timeframe for optimal trade frequency, Donchian for structure,
    # 12h EMA for HTF trend filter, volume for confirmation, ATR for risk management.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate ATR(14) for volatility and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = np.full(n, np.nan)  # for trailing stop
    lowest_low_since_entry = np.full(n, np.nan)
    
    for i in range(50, n):  # start after warmup period
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        # Donchian breakout conditions
        breakout_high = close[i] > donchian_high[i]
        breakout_low = close[i] < donchian_low[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Update trailing stop levels
        if position == 1:  # long position
            if np.isnan(highest_high_since_entry[i-1]):
                highest_high_since_entry[i] = high[i]
            else:
                highest_high_since_entry[i] = max(highest_high_since_entry[i-1], high[i])
        elif position == -1:  # short position
            if np.isnan(lowest_low_since_entry[i-1]):
                lowest_low_since_entry[i] = low[i]
            else:
                lowest_low_since_entry[i] = min(lowest_low_since_entry[i-1], low[i])
        else:
            # Reset trailing levels when flat
            highest_high_since_entry[i] = np.nan
            lowest_low_since_entry[i] = np.nan
        
        # ATR-based trailing stop conditions
        long_stop = (position == 1 and 
                     not np.isnan(highest_high_since_entry[i]) and
                     close[i] < (highest_high_since_entry[i] - 2.0 * atr[i]))
        short_stop = (position == -1 and 
                      not np.isnan(lowest_low_since_entry[i]) and
                      close[i] > (lowest_low_since_entry[i] + 2.0 * atr[i]))
        
        # Entry logic: Donchian breakout + EMA trend + volume confirmation
        long_entry = breakout_high and above_ema and vol_confirm
        short_entry = breakout_low and below_ema and vol_confirm
        
        if long_entry and position != 1:
            position = 1
            highest_high_since_entry[i] = high[i]  # initialize trailing stop
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            lowest_low_since_entry[i] = low[i]  # initialize trailing stop
            signals[i] = -0.25
        elif long_stop or short_stop:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_ema_volume_atr_stop_v1"
timeframe = "4h"
leverage = 1.0