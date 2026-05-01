#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses 12h EMA(50) > 12h EMA(200) as trend filter (bullish/bearish alignment)
# Donchian breakout from prior 4h bar + volume spike (2.0x 20-period MA) for entry
# Works in bull/bear via trend filter - only trades in aligned trends, avoids chop
# Designed for low frequency (75-200 trades over 4 years) to minimize fee drag on 4h timeframe

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) and EMA(200) for trend filter
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Trend: EMA50 > EMA200 (bullish) or EMA50 < EMA200 (bearish)
    ema_trend_bullish = ema_50 > ema_200
    ema_trend_bearish = ema_50 < ema_200
    
    ema_trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_bullish)
    ema_trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_bearish)
    
    # Donchian channels from prior 4h bar (20-period)
    prior_high = np.concatenate([[np.nan], high[:-1]])
    prior_low = np.concatenate([[np.nan], low[:-1]])
    
    donchian_high = pd.Series(prior_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prior_low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(200, 20)  # Need EMA200 and Donchian20
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(ema_trend_bullish_aligned[i]) or 
            np.isnan(ema_trend_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > donchian_high[i]  # Price breaks above Donchian high
        breakout_short = close[i] < donchian_low[i]  # Price breaks below Donchian low
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian high with volume spike and bullish trend
            if breakout_long and vol_spike and ema_trend_bullish_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian low with volume spike and bearish trend
            elif breakout_short and vol_spike and ema_trend_bearish_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below Donchian low or trend weakening
            if close[i] < donchian_low[i] or not ema_trend_bullish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above Donchian high or trend weakening
            if close[i] > donchian_high[i] or not ema_trend_bearish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals