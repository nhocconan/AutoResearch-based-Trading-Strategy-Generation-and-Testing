#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA21 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel AND price > 1w EMA21 AND volume > 1.5x 20-bar average.
# Short when price breaks below lower Donchian channel AND price < 1w EMA21 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 1d timeframe to capture daily trends.
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) via 1w EMA filter.

name = "1d_Donchian20_1wEMA21_VolumeConfirm_v2"
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
    
    # Pre-compute session hours for efficiency (not used for 1d but kept for consistency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # 1w EMA21 calculation
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # 1d Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_21_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema = ema_21_1w_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 1.5)
        
        # Donchian breakout signals
        breakout_up = curr_high > curr_upper  # price breaks above upper channel
        breakout_down = curr_low < curr_lower  # price breaks below lower channel
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish breakout AND price > 1w EMA21 AND volume confirmation
            if (breakout_up and 
                curr_close > curr_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout AND price < 1w EMA21 AND volume confirmation
            elif (breakout_down and 
                  curr_close < curr_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < 1w EMA21 (trend change) OR bearish Donchian breakout (stop and reverse)
            if (curr_close < curr_ema or 
                breakout_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > 1w EMA21 (trend change) OR bullish Donchian breakout (stop and reverse)
            if (curr_close > curr_ema or 
                breakout_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals