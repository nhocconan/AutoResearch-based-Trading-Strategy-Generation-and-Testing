#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and 1d volume confirmation
# - Primary signal: Price breaks above/below Donchian(20) channel on 4h
# - Trend filter: 12h EMA(50) slope positive for longs, negative for shorts (avoids counter-trend)
# - Volume confirmation: 1d volume > 1.5x 20-period average volume (institutional participation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(20) on 4h
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines

name = "4h_12h_1d_donchian_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 60 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(50) for trend filter (using close)
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope = ema_50 - np.roll(ema_50, 1)  # Positive slope = uptrend
    ema_slope[0] = 0
    ema_trend_long = ema_slope > 0   # Uptrend condition
    ema_trend_short = ema_slope < 0  # Downtrend condition
    ema_trend_long_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_long)
    ema_trend_short_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_short)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(20) for stoploss
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_20 = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_trend_long_aligned[i]) or np.isnan(ema_trend_short_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_4h[i] < donchian_low[i] or close_4h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_4h[i] > donchian_high[i] or close_4h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with trend and volume filters
            if vol_spike_aligned[i]:
                # Long: price breaks above Donchian high AND 12h uptrend
                if close_4h[i] > donchian_high[i] and ema_trend_long_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price breaks below Donchian low AND 12h downtrend
                elif close_4h[i] < donchian_low[i] and ema_trend_short_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals