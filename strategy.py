#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses 1w EMA > 1w EMA50 slope to identify strong trends, reducing whipsaws in ranging markets.
# Long when price breaks above Donchian upper (20) AND 1w EMA50 slope > 0 AND volume > 2.0x 20-bar average.
# Short when price breaks below Donchian lower (20) AND 1w EMA50 slope < 0 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Volume spike threshold set to 2.0x to ensure high-quality signals and limit trade frequency.
# Works in both bull and bear markets by only taking trades in the direction of the 1w trend.

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 1w EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA50 slope (positive = uptrend, negative = downtrend)
    # Using 5-period difference to smooth noise
    ema50_slope = np.zeros_like(ema_50)
    ema50_slope[5:] = (ema_50[5:] - ema_50[:-5]) / 5
    
    # Align 1w EMA50 slope to 1d timeframe
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1w, ema50_slope)
    
    # 1w trend: EMA50 slope > 0 for uptrend, < 0 for downtrend
    uptrend = ema50_slope_aligned > 0
    downtrend = ema50_slope_aligned < 0
    
    # Calculate Donchian channels (20-period) on 1d timeframe
    # Upper channel: highest high of last 20 periods
    # Lower channel: lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 1d volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(ema50_slope_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_upper[i]  # break above upper channel
        breakout_down = curr_low < donchian_lower[i]  # break below lower channel
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper AND 1w uptrend AND volume confirmation
            if (breakout_up and 
                uptrend[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower AND 1w downtrend AND volume confirmation
            elif (breakout_down and 
                  downtrend[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower channel (stoploss) OR trend changes
            if (curr_low < donchian_lower[i] or 
                not uptrend[i]):  # trend weakened or reversed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper channel (stoploss) OR trend changes
            if (curr_high > donchian_upper[i] or 
                not downtrend[i]):  # trend weakened or reversed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals