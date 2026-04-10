#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (EMA50) and volume confirmation
# - Long when price breaks above 20-day high in 1w uptrend (close > EMA50) with volume spike (>1.5x 20-day avg volume)
# - Short when price breaks below 20-day low in 1w downtrend (close < EMA50) with volume spike
# - Exit on opposite Donchian breakout (mean reversion) or ATR-based stoploss (2.0x ATR(14))
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets: trend filter ensures we only trade with the 1w trend,
#   while Donchian breakouts capture momentum; volume confirmation avoids false breakouts

name = "1d_1w_donchian_breakout_volume_trend_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    
    # 1d Donchian channels (20-period)
    # Upper channel = 20-period high, Lower channel = 20-period low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1d[i]) or 
            np.isnan(atr_14_1d[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price breaks below Donchian low (mean reversion)
            if (close_1d[i] < entry_price - 2.0 * entry_atr or 
                close_1d[i] < donchian_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price breaks above Donchian high (mean reversion)
            if (close_1d[i] > entry_price + 2.0 * entry_atr or 
                close_1d[i] > donchian_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike_1d[i]:
                # Long signal: price breaks above Donchian high in 1w uptrend
                if (close_1d[i] > donchian_high[i] and 
                    close_1d[i] > ema_50_1w_aligned[i]):
                    position = 1
                    entry_price = close_1d[i]
                    entry_atr = atr_14_1d[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian low in 1w downtrend
                elif (close_1d[i] < donchian_low[i] and 
                      close_1d[i] < ema_50_1w_aligned[i]):
                    position = -1
                    entry_price = close_1d[i]
                    entry_atr = atr_14_1d[i]
                    signals[i] = -0.25
    
    return signals