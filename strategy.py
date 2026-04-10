#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume spike confirmation
# - Williams %R(14) from 6h: < -80 = oversold, > -20 = overbought
# - 12h EMA(50) trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
# - Volume confirmation: current volume > 2.0x 20-period average
# - Long signals: Williams %R < -80 AND price > 12h EMA50 AND volume spike (buy oversold in uptrend)
# - Short signals: Williams %R > -20 AND price < 12h EMA50 AND volume spike (sell overbought in downtrend)
# - Exit: Williams %R crosses back above -50 (for long) or below -50 (for short) OR stoploss
# - Designed for 6h timeframe: targets 12-30 trades/year to avoid fee drag
# - Works in bull/bear markets: trend filter ensures we trade with higher timeframe momentum

name = "6h_12h_williamsr_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low + 1e-10)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (2.0 * avg_volume_20)
    
    # Pre-compute 6h ATR(14) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R > -50 (mean reversion) OR stoploss hit
            if williams_r[i] > -50 or close_6h[i] < entry_price - 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (mean reversion) OR stoploss hit
            if williams_r[i] < -50 or close_6h[i] > entry_price + 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R signals with trend and volume filters
            if vol_spike[i]:
                # Long: oversold in uptrend
                if williams_r[i] < -80 and close_6h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: overbought in downtrend
                elif williams_r[i] > -20 and close_6h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals