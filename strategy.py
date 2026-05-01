#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1w EMA50 as trend filter (bullish when price > EMA50, bearish when price < EMA50)
# Donchian levels from prior 1d bar provide precise breakout zones
# Volume spike (2x 20-period MA) confirms institutional participation
# Designed for low frequency (30-100 trades over 4 years) to minimize fee drag
# Works in bull/bear via trend filter + price structure logic

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
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
    
    # 1w HTF data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation (trend filter)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Donchian levels from prior 1d bar (using prior bar's HL)
    # Donchian: Upper = prior 20-period high, Lower = prior 20-period low
    prior_high = np.concatenate([[np.nan], high[:-1]])  # prior bar's high
    prior_low = np.concatenate([[np.nan], low[:-1]])    # prior bar's low
    
    donchian_upper = pd.Series(prior_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(prior_low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20, 20)  # Need 1w EMA50, Donchian20, and volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > donchian_upper[i]  # Price breaks above Donchian upper
        breakout_short = close[i] < donchian_lower[i]  # Price breaks below Donchian lower
        
        # Trend filter: price > EMA50 for long, price < EMA50 for short
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian upper with volume spike and bullish trend
            if breakout_long and vol_spike and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian lower with volume spike and bearish trend
            elif breakout_short and vol_spike and bearish_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below Donchian lower or trend turning bearish
            if close[i] < donchian_lower[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above Donchian upper or trend turning bullish
            if close[i] > donchian_upper[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals