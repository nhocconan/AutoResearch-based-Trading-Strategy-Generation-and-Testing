#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band squeeze breakout with 1-day trend filter (EMA34) and volume confirmation.
# Long when: Bollinger Band width < 20th percentile (squeeze) AND price breaks above upper band AND EMA34(1d) rising AND volume > 1.5 * EMA20(volume).
# Short when: Bollinger Band width < 20th percentile (squeeze) AND price breaks below lower band AND EMA34(1d) falling AND volume > 1.5 * EMA20(volume).
# Exit when price crosses back inside the Bollinger Bands.
# Bollinger squeeze identifies low volatility periods preceding breakouts; EMA34 filters trend direction; volume confirms breakout strength.
# Designed for low trade frequency (target: 20-40/year) to minimize fee drag and improve generalization in both bull and bear markets.
name = "4h_BollingerSqueeze_1dEMA34_VolumeBreakout"
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
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + bb_std * std_20
    lower_band = sma_20 - bb_std * std_20
    bb_width = upper_band - lower_band
    
    # Bollinger Band width percentile (20th percentile lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # EMA34 on 1d for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(squeeze_condition[i]) or np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze AND price breaks above upper band AND EMA34(1d) rising AND volume confirmation
            long_condition = squeeze_condition[i] and (close[i] > upper_band[i]) and ema_34_rising_aligned[i] and volume_confirm[i]
            # Short: squeeze AND price breaks below lower band AND EMA34(1d) falling AND volume confirmation
            short_condition = squeeze_condition[i] and (close[i] < lower_band[i]) and ema_34_falling_aligned[i] and volume_confirm[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses back inside Bollinger Bands (below upper band)
            if close[i] < upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses back inside Bollinger Bands (above lower band)
            if close[i] > lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals