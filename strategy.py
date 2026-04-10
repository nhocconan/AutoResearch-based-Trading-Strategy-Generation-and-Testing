#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# - Long: price breaks above Donchian(20) high AND close > 1w EMA50 (uptrend)
# - Short: price breaks below Donchian(20) low AND close < 1w EMA50 (downtrend)
# - Volume confirmation: 1d volume > 1.5x 20-period average
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.5x ATR(20) on 1d
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Trend filter avoids counter-trend trades; volume confirmation reduces false breakouts

name = "1d_1w_donchian_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Pre-compute 1d Donchian channels
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume spike filter
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    
    # Pre-compute 1d ATR(20) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR stoploss hit
            if close_1d[i] < donchian_low[i] or close_1d[i] < entry_price - 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR stoploss hit
            if close_1d[i] > donchian_high[i] or close_1d[i] > entry_price + 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Long: price breaks above Donchian high in uptrend (close > EMA50)
                if close_1d[i] > donchian_high[i] and close_1d[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_1d[i]
                    signals[i] = 0.25
                # Short: price breaks below Donchian low in downtrend (close < EMA50)
                elif close_1d[i] < donchian_low[i] and close_1d[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_1d[i]
                    signals[i] = -0.25
    
    return signals