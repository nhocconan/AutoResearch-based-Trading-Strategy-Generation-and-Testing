#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND weekly EMA34 is rising AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian(20) low AND weekly EMA34 is falling AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Weekly EMA34 filter ensures alignment with higher timeframe momentum, reducing false breakouts.
# Volume confirmation set to 1.5x to avoid choppy market noise while capturing institutional participation.
# Primary timeframe: 1d, HTF: 1w for weekly bias.

name = "1d_Donchian20_WeeklyEMA34_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Weekly EMA34: calculate on weekly close, check if rising/falling
    weekly_close = df_1w['close'].values
    ema_34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Weekly EMA34 trend: 1 = rising (current > previous), -1 = falling (current < previous), 0 = flat
    weekly_ema_trend_raw = np.where(ema_34[1:] > ema_34[:-1], 1,
                                    np.where(ema_34[1:] < ema_34[:-1], -1, 0))
    # Prepend 0 for first value (no previous to compare)
    weekly_ema_trend_raw = np.concatenate([[0], weekly_ema_trend_raw])
    weekly_ema_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_trend_raw)
    
    # Calculate Donchian(20) channels from 1d data (structure from higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian high: max(high, 20) from previous completed day
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    # Donchian low: min(low, 20) from previous completed day
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (should be 1:1 but using align for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: current 1d volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(weekly_ema_trend_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume confirmation threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high_aligned[i]  # break above Donchian high
        breakout_down = curr_low < donchian_low_aligned[i]  # break below Donchian low
        
        # Weekly EMA34 trend filter
        weekly_uptrend = weekly_ema_trend_aligned[i] == 1
        weekly_downtrend = weekly_ema_trend_aligned[i] == -1
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND weekly uptrend AND volume confirmation
            if (breakout_up and 
                weekly_uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND weekly downtrend AND volume confirmation
            elif (breakout_down and 
                  weekly_downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR weekly trend turns down
            if (curr_low < donchian_low_aligned[i] or 
                weekly_ema_trend_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR weekly trend turns up
            if (curr_high > donchian_high_aligned[i] or 
                weekly_ema_trend_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals