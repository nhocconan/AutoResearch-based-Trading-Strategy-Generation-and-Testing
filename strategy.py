#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 4-hour ADX(14) trend strength and 1-hour Bollinger Band(20,2) mean reversion.
# In trending markets (ADX > 25), trade breakouts in direction of 4h trend.
# In ranging markets (ADX <= 25), trade reversals at Bollinger Bands.
# This adapts to both bull and bear markets by using trend strength filter.
# Volume > 1.3x 20-period average confirms momentum.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h ADX(14) for trend strength
    adx_len = 14
    if len(df_4h) < adx_len:
        return np.zeros(n)
    
    # Calculate ADX components
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Bollinger Bands (20, 2) on 1h
    bb_len = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_len, min_periods=bb_len).mean().values
    std = pd.Series(close).rolling(window=bb_len, min_periods=bb_len).std().values
    bb_upper = sma + bb_std * std
    bb_lower = sma - bb_std * std
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(adx_len*2, bb_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(sma[i]) or
            np.isnan(std[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend regime: ADX > 25 = trending, ADX <= 25 = ranging
        trending = adx_aligned[i] > 25
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            if trending:
                # In trending market: trade breakouts in direction of 4h trend
                # Need 4h trend direction - use price vs 4h EMA20 as proxy
                if len(df_4h) >= 20:
                    ema_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
                    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
                    if not np.isnan(ema_4h_aligned[i]):
                        bullish_4h = close[i] > ema_4h_aligned[i]
                        bearish_4h = close[i] < ema_4h_aligned[i]
                        
                        # Breakout above upper band in bullish 4h trend
                        if (close[i] > bb_upper[i] and 
                            bullish_4h and 
                            volume_confirmed):
                            position = 1
                            signals[i] = position_size
                        # Breakdown below lower band in bearish 4h trend
                        elif (close[i] < bb_lower[i] and 
                              bearish_4h and 
                              volume_confirmed):
                            position = -1
                            signals[i] = -position_size
                        else:
                            signals[i] = 0.0
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            else:
                # In ranging market: trade reversals at Bollinger Bands
                # Long at lower band, short at upper band
                if (close[i] < bb_lower[i] and 
                    volume_confirmed):
                    position = 1
                    signals[i] = position_size
                elif (close[i] > bb_upper[i] and 
                      volume_confirmed):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle (SMA) or breaks below lower band with volume
            if (close[i] > sma[i] or 
                (close[i] < bb_lower[i] and volume[i] < vol_ma[i] * 0.5)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle (SMA) or breaks above upper band with low volume
            if (close[i] < sma[i] or 
                (close[i] > bb_upper[i] and volume[i] < vol_ma[i] * 0.5)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_ADX_BB_MeanRev_TrendAdapt_v1"
timeframe = "1h"
leverage = 1.0