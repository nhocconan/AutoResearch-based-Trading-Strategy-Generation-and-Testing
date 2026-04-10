#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1d EMA50 > EMA200 (bullish trend) AND 4h volume > 1.5x 20-period volume SMA
# - Short when price breaks below Donchian(20) low AND 1d EMA50 < EMA200 (bearish trend) AND 4h volume > 1.5x 20-period volume SMA
# - Exit: opposite Donchian breakout or volume drops below average
# - Uses 4h for Donchian and volume, 1d for EMA trend filter
# - EMA trend filter ensures we trade with the higher timeframe trend
# - Volume confirmation ensures breakouts have conviction
# - Donchian breakouts capture sustained moves in both bull and bear markets
# - Target: 20-50 trades/year to minimize fee drag while capturing meaningful moves

name = "4h_1d_ema_trend_donchian_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 1d EMAs for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMAs to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Pre-compute Donchian channels for 4h data (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume SMA for 4h data (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout signals (using prior bar's channel to avoid look-ahead)
        breakout_long = close[i] > donchian_high[i-1]  # Break above prior period's high
        breakout_short = close[i] < donchian_low[i-1]  # Break below prior period's low
        
        # 1d EMA trend filter
        bullish_trend = ema_50_aligned[i] > ema_200_aligned[i]
        bearish_trend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Exit conditions: opposite breakout or volume drops below average
        exit_long = close[i] < donchian_low[i-1] or volume[i] < volume_sma_20[i]
        exit_short = close[i] > donchian_high[i-1] or volume[i] < volume_sma_20[i]
        
        # Trading logic
        if vol_confirm:
            # Long: Donchian breakout above with bullish 1d trend
            if breakout_long and bullish_trend:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Donchian breakout below with bearish 1d trend
            elif breakout_short and bearish_trend:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals