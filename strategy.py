#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 as trend filter (bullish when price > EMA50, bearish when price < EMA50)
# Donchian channels from prior 4h bar provide precise breakout zones
# Volume spike (2x 20-period MA) confirms institutional participation
# Designed for low frequency (75-200 trades over 4 years) to minimize fee drag
# Works in bull/bear via trend filter + price structure logic

name = "4h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation (trend filter)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels from prior 4h bar (using prior bar's HL)
    # Upper = max(high of last 20 periods), Lower = min(low of last 20 periods)
    prior_high = np.concatenate([[np.nan], high[:-1]])  # prior bar's high
    prior_low = np.concatenate([[np.nan], low[:-1]])    # prior bar's low
    
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(prior_high)
    low_series = pd.Series(prior_low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20, 20)  # Need 1d EMA50, Donchian(20), and volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > donchian_upper[i]  # Price breaks above upper Donchian
        breakout_short = close[i] < donchian_lower[i]  # Price breaks below lower Donchian
        
        # Trend filter: price > EMA50 for long, price < EMA50 for short
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian with volume spike and bullish trend
            if breakout_long and vol_spike and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower Donchian with volume spike and bearish trend
            elif breakout_short and vol_spike and bearish_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below prior bar's low or trend turning bearish
            if close[i] < prior_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above prior bar's high or trend turning bullish
            if close[i] > prior_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals