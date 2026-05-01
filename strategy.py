#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 1d EMA50 trend filter and 12h volume spike confirmation.
# Uses 1d EMA > EMA50 for bullish bias, EMA < EMA50 for bearish bias to align with higher timeframe trend.
# Long when price breaks above Donchian(20) high AND 1d EMA > EMA50 AND 12h volume > 1.5x 20-bar average.
# Short when price breaks below Donchian(20) low AND 1d EMA < EMA50 AND 12h volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years.
# Volume confirmation uses 12h timeframe to avoid noise on 6h bars while maintaining responsiveness.

name = "6h_Donchian20_1dEMA50_Trend_12hVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d trend: EMA > EMA50 = bullish, EMA < EMA50 = bearish
    ema_close_1d = df_1d['close'].values
    ema_50_check = pd.Series(ema_close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    bullish_trend = ema_close_1d > ema_50_check
    bearish_trend = ema_close_1d < ema_50_check
    
    # Align trend signals to 6h timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend)
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need enough for volume MA
        return np.zeros(n)
    
    # 12h volume MA calculation
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h volume MA to 6h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Using rolling window with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        
        # Volume confirmation: current 6h volume > 1.5x 12h volume MA (scaled)
        # Scale 12h MA to 6h by assuming ~2x volume per 6h bar vs 12h bar
        vol_ma_scaled = vol_ma_12h_aligned[i] * 0.5  # Approximate 6h equivalent
        if vol_ma_scaled <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (vol_ma_scaled * 1.5)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above upper channel
        breakout_down = curr_low < donchian_low[i]  # break below lower channel
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND 1d bullish trend AND volume confirmation
            if (breakout_up and 
                bullish_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND 1d bearish trend AND volume confirmation
            elif (breakout_down and 
                  bearish_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR trend turns bearish
            if (curr_low < donchian_low[i] or 
                bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR trend turns bullish
            if (curr_high > donchian_high[i] or 
                bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals