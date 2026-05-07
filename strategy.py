#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Bollinger Band squeeze breakout with 1-day trend filter and volume confirmation.
# Long when: Bollinger Band width < 5th percentile (squeeze) AND close > upper BB AND 1-day EMA34 rising AND volume > 1.3 * 20-period EMA(volume).
# Short when: Bollinger Band width < 5th percentile (squeeze) AND close < lower BB AND 1-day EMA34 falling AND volume > 1.3 * 20-period EMA(volume).
# Exit when price crosses back inside the Bollinger Bands.
# Designed for low trade frequency (target: 15-30/year) to minimize fee drag and improve generalization.
# Works in bull markets via upward breakouts from squeeze and in bear markets via downward breakouts from squeeze.
name = "6h_BollingerSqueeze_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Bollinger Band width percentile (5th percentile lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.05).values
    squeeze_condition = bb_width <= bb_width_percentile
    
    # Bollinger Band breakout conditions
    bb_breakout_up = close > upper_bb
    bb_breakout_down = close < lower_bb
    
    # Load 1-day data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA34 on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Rising if current > previous, falling if current < previous
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: current volume > 1.3 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(ema_34_rising_aligned[i]) or 
            np.isnan(ema_34_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Squeeze AND breakout up AND 1-day EMA34 rising AND volume spike
            long_condition = squeeze_condition[i] and bb_breakout_up[i] and ema_34_rising_aligned[i] and volume_spike[i]
            # Short: Squeeze AND breakout down AND 1-day EMA34 falling AND volume spike
            short_condition = squeeze_condition[i] and bb_breakout_down[i] and ema_34_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses back inside Bollinger Bands (close < upper AND close > lower)
            if close[i] < upper_bb[i] and close[i] > lower_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses back inside Bollinger Bands (close < upper AND close > lower)
            if close[i] < upper_bb[i] and close[i] > lower_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals