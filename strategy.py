#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high with 1w uptrend (close > EMA50) and volume spike
# - Short when price breaks below Donchian(20) low with 1w downtrend (close < EMA50) and volume spike
# - Uses 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# - 1w EMA50 filter ensures trading with higher timeframe trend direction
# - 12h volume > 2.0x 20-period average confirms breakout strength
# - Discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) or Donchian breakout in opposite direction

name = "12h_1w_donchian_volume_trend_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w volume confirmation: > 2.0x 20-period average
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (2.0 * avg_volume_20_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # 1w ATR(14) for stoploss
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1w = np.zeros_like(tr)
    atr_14_1w[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1w[i] = (atr_14_1w[i-1] * (14-1) + tr[i]) / 14
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # 12h Donchian(20)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h ATR(14) for stoploss
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_14_12h = np.zeros_like(tr_12h)
    atr_14_12h[14-1] = np.mean(tr_12h[:14])
    for i in range(14, len(tr_12h)):
        atr_14_12h[i] = (atr_14_12h[i-1] * (14-1) + tr_12h[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or Donchian breakout below low
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] < donchian_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or Donchian breakout above high
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] > donchian_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike_1w_aligned[i]:
                # Long signal: breakout above Donchian high in 1w uptrend
                if prices['close'].iloc[i] > donchian_high[i] and prices['close'].iloc[i] > ema_50_1w_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_12h[i]
                    signals[i] = 0.25
                # Short signal: breakout below Donchian low in 1w downtrend
                elif prices['close'].iloc[i] < donchian_low[i] and prices['close'].iloc[i] < ema_50_1w_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_12h[i]
                    signals[i] = -0.25
    
    return signals