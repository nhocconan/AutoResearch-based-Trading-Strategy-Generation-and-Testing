#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Uses 6h primary timeframe for optimal trade frequency (target: 12-37/year)
# 1d ATR regime filter distinguishes trending (ATR(7)/ATR(30) > 1.2) from ranging markets
# In trending regime: trade Donchian breakouts in direction of 1d EMA50 trend
# In ranging regime: fade Donchian touches at bands with mean reversion
# Volume confirmation (>1.3 * 20-period EMA) ensures participation on signals
# Designed for low trade frequency with 0.25 sizing to manage drawdown
# Works in bull markets via trend-following breakouts and bear markets via mean reversion in ranges

name = "6h_Donchian20_1dATRRegime_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ATR regime and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(7) and ATR(30) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_7 = pd.Series(tr).ewm(span=7, adjust=False, min_periods=7).mean().values
    atr_30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # ATR ratio for regime: >1.2 = trending, <1.2 = ranging
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    # Calculate 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Using rolling window with min_periods
    df_prices = pd.DataFrame({'high': high, 'low': low, 'close': close})
    donchian_high = df_prices['high'].rolling(window=20, min_periods=20).max().values
    donchian_low = df_prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.3 * 20-period EMA (6h * ~5.3 = 20 periods)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        atr_ratio_val = atr_ratio_aligned[i]
        is_trending = atr_ratio_val > 1.2
        is_ranging = atr_ratio_val <= 1.2
        
        if position == 0:  # Flat - look for new entries
            if is_trending:
                # Trending regime: trade Donchian breakouts in direction of 1d EMA50 trend
                bullish_trend = close[i] > ema_50_1d_aligned[i]
                bearish_trend = close[i] < ema_50_1d_aligned[i]
                
                if bullish_trend:
                    # Long: price breaks above Donchian high with volume spike
                    if close[i] > donchian_high[i] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                    else:
                        signals[i] = 0.0
                elif bearish_trend:
                    # Short: price breaks below Donchian low with volume spike
                    if close[i] < donchian_low[i] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Avoid chop around 1d EMA50
            else:
                # Ranging regime: fade Donchian touches at bands
                # Long: price touches or pierces Donchian low with volume spike (mean reversion up)
                if close[i] <= donchian_low[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price touches or pierces Donchian high with volume spike (mean reversion down)
                elif close[i] >= donchian_high[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            if is_trending:
                # In trending regime: exit on Donchian low break or trend reversal
                if close[i] < donchian_low[i] or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging regime: exit on Donchian high touch (mean reversion target)
                if close[i] >= donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:  # Short position
            if is_trending:
                # In trending regime: exit on Donchian high break or trend reversal
                if close[i] > donchian_high[i] or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging regime: exit on Donchian low touch (mean reversion target)
                if close[i] <= donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals