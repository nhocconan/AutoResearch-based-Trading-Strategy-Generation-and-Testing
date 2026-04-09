#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d/1w regime filter
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# In bull markets (price > weekly EMA50): go long when Bull Power > 0 and rising
# In bear markets (price < weekly EMA50): go short when Bear Power < 0 and falling
# In ranging markets (price near weekly EMA50): mean revert at 6h Bollinger Bands (20,2)
# Volume confirmation: current 6h volume > 1.5x daily average volume
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_1w_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for regime filter
    close_1w = df_1w['close'].values
    close_s_1w = pd.Series(close_1w)
    ema50_1w = close_s_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Bollinger Bands (20,2) for ranging regime
    sma20 = close_s.rolling(window=20, min_periods=20).mean().values
    std20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Align HTF indicators to 6h timeframe
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray components
        bull_power = high[i] - ema13[i]
        bear_power = low[i] - ema13[i]
        
        # Volume confirmation: current 6h volume > 1.5x daily average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Regime determination from weekly EMA50
        price_vs_weekly_ema = close[i] - ema50_1w_aligned[i]
        weekly_ema_level = ema50_1w_aligned[i]
        bull_market = price_vs_weekly_ema > 0.02 * weekly_ema_level  # >2% above weekly EMA50
        bear_market = price_vs_weekly_ema < -0.02 * weekly_ema_level  # >2% below weekly EMA50
        ranging_market = abs(price_vs_weekly_ema) <= 0.02 * weekly_ema_level  # within 2% of weekly EMA50
        
        if position == 1:  # Long position
            # Exit: Bull Power turns negative OR regime shifts to bear/bearish
            if bull_power <= 0 or bear_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR regime shifts to bull/bullish
            if bear_power >= 0 or bull_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if bull_market and volume_confirmed:
                # Bull market: go long when Bull Power > 0 and rising (momentum)
                if bull_power > 0 and bull_power > (high[i-1] - ema13[i-1]) if i > 0 else bull_power > 0:
                    position = 1
                    signals[i] = 0.25
            elif bear_market and volume_confirmed:
                # Bear market: go short when Bear Power < 0 and falling (momentum)
                if bear_power < 0 and bear_power < (low[i-1] - ema13[i-1]) if i > 0 else bear_power < 0:
                    position = -1
                    signals[i] = -0.25
            elif ranging_market and volume_confirmed:
                # Ranging market: mean revert at Bollinger Bands
                if close[i] < lower_bb[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > upper_bb[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals