# [Experiment 44310] Hypothesis: 1d timeframe strategy using weekly Donchian breakouts with volume confirmation and volatility filter. Designed for low trade frequency to avoid fee drag while capturing major trends in both bull and bear markets. Uses 20-period weekly Donchian channels with volume spike confirmation (>2x average) and ATR-based volatility filter to avoid choppy markets.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    volume_weekly = df_weekly['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    upper_channel = np.full_like(high_weekly, np.nan)
    lower_channel = np.full_like(low_weekly, np.nan)
    
    for i in range(19, len(high_weekly)):
        upper_channel[i] = np.max(high_weekly[i-19:i+1])
        lower_channel[i] = np.min(low_weekly[i-19:i+1])
    
    # Align weekly channels to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower_channel)
    
    # Calculate weekly ATR for volatility filter (14-period)
    tr_weekly = np.zeros(len(df_weekly))
    tr_weekly[0] = high_weekly[0] - low_weekly[0]
    for i in range(1, len(df_weekly)):
        tr_weekly[i] = max(
            high_weekly[i] - low_weekly[i],
            abs(high_weekly[i] - close_weekly[i-1]),
            abs(low_weekly[i] - close_weekly[i-1])
        )
    
    atr_weekly = np.full(len(df_weekly), np.nan)
    if len(df_weekly) >= 14:
        atr_weekly[13] = np.mean(tr_weekly[:14])
        for i in range(14, len(df_weekly)):
            atr_weekly[i] = (atr_weekly[i-1] * 13 + tr_weekly[i]) / 14
    
    atr_aligned = align_htf_to_ltf(prices, df_weekly, atr_weekly)
    
    # Volume spike detection (20-period average on weekly)
    vol_ma_weekly = np.full_like(volume_weekly, np.nan)
    if len(volume_weekly) >= 20:
        for i in range(19, len(volume_weekly)):
            vol_ma_weekly[i] = np.mean(volume_weekly[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_weekly, vol_ma_weekly)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or
            np.isnan(atr_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility periods
        if atr_aligned[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current weekly volume vs 20-period average
        if vol_ma_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume_weekly[-1] / vol_ma_aligned[i] if len(volume_weekly) > 0 else 0
        # Use current weekly volume for ratio calculation
        current_weekly_vol = volume_weekly[-1] if len(volume_weekly) > 0 else 0
        if len(volume_weekly) >= 20 and vol_ma_aligned[i] > 0:
            volume_ratio = current_weekly_vol / vol_ma_aligned[i]
        else:
            volume_ratio = 0
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above upper weekly Donchian with volume confirmation
            if close[i] > upper_aligned[i] and volume_ratio > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower weekly Donchian with volume confirmation
            elif close[i] < lower_aligned[i] and volume_ratio > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below lower weekly Donchian
            if close[i] < lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above upper weekly Donchian
            if close[i] > upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Breakout_Volume_Filter"
timeframe = "1d"
leverage = 1.0