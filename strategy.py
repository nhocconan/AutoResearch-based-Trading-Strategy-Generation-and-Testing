#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper band with 1w uptrend (close > 1w EMA34) and volume > 2.0x 20-bar avg.
# Short when price breaks below Donchian lower band with 1w downtrend (close < 1w EMA34) and volume > 2.0x 20-bar avg.
# Exit on opposite Donchian band touch (mean reversion within the channel).
# Uses proven Donchian structure with strict volume confirmation (2.0x) and 1w EMA34 trend filter to limit trades (target 7-25/year).
# 1w EMA34 provides higher timeframe trend filter to reduce false signals in choppy markets and avoid SOL-only bias.
# Timeframe: 1d, HTF: 1w as per experiment guidelines.

name = "1d_Donchian20_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Previous 1w OHLC for completed 1w bar (no look-ahead)
    df_1w_prev = get_htf_data(prices, '1w')
    if len(df_1w_prev) < 2:
        return np.zeros(n)
    
    prev_high_1w = df_1w_prev['high'].shift(1).values
    prev_low_1w = df_1w_prev['low'].shift(1).values
    prev_close_1w = df_1w_prev['close'].shift(1).values
    
    # Align 1w data to 1d timeframe (completed 1w bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_1w_prev, prev_high_1w)
    prev_low_aligned = align_htf_to_ltf(prices, df_1w_prev, prev_low_1w)
    prev_close_aligned = align_htf_to_ltf(prices, df_1w_prev, prev_close_1w)
    
    # Donchian(20) from previous completed 1w bar (no look-ahead)
    donchian_high = pd.Series(prev_high_aligned).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low_aligned).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_donch_high = donchian_high[i]
        curr_donch_low = donchian_low[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, uptrend (close > 1w EMA34), volume spike
            if (curr_close > curr_donch_high and 
                curr_close > curr_ema_34_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, downtrend (close < 1w EMA34), volume spike
            elif (curr_close < curr_donch_low and 
                  curr_close < curr_ema_34_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Donchian low (mean reversion)
            if curr_close <= curr_donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches Donchian high (mean reversion)
            if curr_close >= curr_donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals