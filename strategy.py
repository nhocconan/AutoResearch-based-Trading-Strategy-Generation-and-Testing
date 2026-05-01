#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze Breakout with 1d EMA50 trend filter and volume confirmation
# Bollinger Squeeze identifies low volatility periods (band width < 20th percentile) preceding explosive moves
# Breakout occurs when price closes outside Bollinger Bands after squeeze + 1d EMA50 trend alignment + volume spike
# Designed to capture explosive moves in both bull and bear markets with tight entry conditions to minimize fee drag
# Target: 20-50 trades/year (80-200 over 4 years) to avoid overtrading and fee drag

name = "4h_BollingerSqueeze_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    bb_width = (upper_band - lower_band) / sma_20  # Normalized band width
    
    # Bollinger Squeeze: band width < 20th percentile of last 50 periods
    bb_width_ma_50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need sufficient history for 1d EMA, BB, volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger Breakout conditions
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Squeeze must be active (low volatility) before breakout
        is_squeezed = squeeze_condition[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish breakout after squeeze, volume spike, uptrend
            if breakout_up and is_squeezed and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish breakout after squeeze, volume spike, downtrend
            elif breakout_down and is_squeezed and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish breakout or trend reversal
            if breakout_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish breakout or trend reversal
            if breakout_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals