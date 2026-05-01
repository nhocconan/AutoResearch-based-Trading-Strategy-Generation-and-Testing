#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses 4h Donchian breakouts for entry timing, filtered by 1d EMA34 trend direction (long above EMA34, short below)
# Volume confirmation > 1.8x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~15-35 trades/year per symbol with 0.20 sizing
# Uses session filter (08-20 UTC) to avoid low-liquidity periods
# Combines multiple timeframes: 1d for trend, 4h for structure, 1h for precise entry timing

name = "1h_Donchian20_1dEMA34_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h HTF data for Donchian(20) structure
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # 1h volume confirmation: volume > 1.8 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d EMA34 (34 bars) + 4h Donchian20 (20 bars) + 1h volume EMA20 (20 bars)
    # Plus extra for alignment safety
    start_idx = max(40, 30, 20)
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any indicator is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_4h_aligned[i]) or 
            np.isnan(donchian_low_4h_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA34: long above EMA34, short below EMA34
        bullish_trend = close[i] > ema_34_1d_aligned[i]
        bearish_trend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_trend:
                # Long: 4h Donchian breakout above upper band with volume spike
                if close[i] > donchian_high_4h_aligned[i] and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_trend:
                # Short: 4h Donchian breakdown below lower band with volume spike
                if close[i] < donchian_low_4h_aligned[i] and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop exactly at EMA34
        
        elif position == 1:  # Long position
            # Exit: 4h Donchian breakdown below lower band (failure of breakout)
            if close[i] < donchian_low_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: 4h Donchian breakout above upper band (failure of breakdown)
            if close[i] > donchian_high_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals