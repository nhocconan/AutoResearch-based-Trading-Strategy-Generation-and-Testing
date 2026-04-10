#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d trend filter + volume confirmation
# - Primary signal: Donchian channel breakout on 12h timeframe
# - Trend filter: 1d close > EMA(50) for longs, < EMA(50) for shorts (institutional trend alignment)
# - Volume filter: 12h volume > 1.8x 30-period average volume (strong momentum confirmation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.5x ATR(20) on 12h (wider stop for lower timeframe noise)
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts; trend filter avoids counter-trend trades in bear markets

name = "12h_1d_donchian_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Donchian upper and lower bands
    upper_donch = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_donch = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h volume spike filter
    avg_volume_30 = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume_12h > (1.8 * avg_volume_30)
    
    # Pre-compute 12h ATR(20) for stoploss
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian band OR stoploss hit
            if close_12h[i] < lower_donch[i] or close_12h[i] < entry_price - 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian band OR stoploss hit
            if close_12h[i] > upper_donch[i] or close_12h[i] > entry_price + 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Long: price breaks above upper Donchian band in uptrend (close > EMA50)
                if close_12h[i] > upper_donch[i] and close_12h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian band in downtrend (close < EMA50)
                elif close_12h[i] < lower_donch[i] and close_12h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals