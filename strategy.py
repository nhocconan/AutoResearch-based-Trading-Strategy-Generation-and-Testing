#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for higher timeframe trend (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on weekly timeframe for trend filter
    if len(close_1w) >= 50:
        close_1w_series = pd.Series(close_1w)
        ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # Load daily data for price channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period) on daily timeframe
    if len(high_1d) >= 20:
        # Upper channel: highest high over 20 days
        high_20 = np.full_like(high_1d, np.nan)
        for i in range(19, len(high_1d)):
            high_20[i] = np.max(high_1d[i-19:i+1])
        
        # Lower channel: lowest low over 20 days
        low_20 = np.full_like(low_1d, np.nan)
        for i in range(19, len(low_1d)):
            low_20[i] = np.min(low_1d[i-19:i+1])
        
        # Align to 1d timeframe (no additional delay needed for breakout)
        donch_high_1d = align_htf_to_ltf(prices, df_1d, high_20)
        donch_low_1d = align_htf_to_ltf(prices, df_1d, low_20)
    else:
        donch_high_1d = np.full(n, np.nan)
        donch_low_1d = np.full(n, np.nan)
    
    # Calculate 20-period average volume on daily timeframe
    if len(volume_1d) >= 20:
        vol_ma_20_1d = np.full_like(volume_1d, np.nan)
        for i in range(19, len(volume_1d)):
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
        vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    else:
        vol_ma_20_1d_aligned = np.full(n, np.nan)
    
    # Calculate 14-period ATR on daily timeframe for volatility filter and stop
    if len(high_1d) >= 14:
        tr = np.zeros(len(df_1d))
        tr[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(df_1d)):
            tr[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
        
        atr_1d = np.full(len(df_1d), np.nan)
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
        
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    else:
        atr_1d_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size to control drawdown
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_1d[i]) or 
            np.isnan(donch_low_1d[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip extremely low volatility periods (ATR < 0.5% of price)
        if atr_1d_aligned[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-period average
        if vol_ma_20_1d_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume_1d[i] / vol_ma_20_1d_aligned[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume confirmation and weekly uptrend
            if (close[i] > donch_high_1d[i] and 
                volume_ratio > vol_threshold and 
                close[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian low with volume confirmation and weekly downtrend
            elif (close[i] < donch_low_1d[i] and 
                  volume_ratio > vol_threshold and 
                  close[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below Donchian low or weekly trend turns bearish
            if (close[i] < donch_low_1d[i] or 
                close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above Donchian high or weekly trend turns bullish
            if (close[i] > donch_high_1d[i] or 
                close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian20_WeeklyEMA50_Volume_Trend"
timeframe = "1d"
leverage = 1.0