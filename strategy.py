#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation (>1.8x 20-bar volume MA) and 1w EMA34 trend filter
# Donchian channels on daily timeframe provide robust structure for capturing major trends.
# Volume spike confirms institutional participation in breakouts.
# 1w EMA34 ensures we only trade in the direction of the weekly trend, reducing false breakouts.
# Discrete sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).
# Works in bull (breakouts with volume) and bear (trend continuation after pullbacks to channel).

name = "1d_Donchian20_Breakout_VolumeSpike_1wEMA34_Trend_v1"
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
    
    # 1w HTF data for EMA34 calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 calculation
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Donchian(20) channels on 1d data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 55  # Need 34 for EMA + 20 for Donchian + volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Donchian breakout conditions (using prior bar channels to avoid look-ahead)
        breakout_up = curr_close > donchian_high[i-1]  # Break above upper channel
        breakout_down = curr_close < donchian_low[i-1]  # Break below lower channel
        
        # Volume confirmation and trend filter
        vol_spike = volume_spike[i]
        # For longs: price above weekly EMA34 (bullish trend)
        # For shorts: price below weekly EMA34 (bearish trend)
        bullish_trend = curr_close > ema_34_1w_aligned[i]
        bearish_trend = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up, volume spike, bullish weekly trend
            if breakout_up and vol_spike and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, volume spike, bearish weekly trend
            elif breakout_down and vol_spike and bearish_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown or trend reversal
            if curr_close < donchian_low[i] or curr_close < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout or trend reversal
            if curr_close > donchian_high[i] or curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals