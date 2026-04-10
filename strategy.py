#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume confirmation and 1w trend filter
# - Primary signal: Williams %R(14) crosses above/below extreme levels on 12h
# - Volume filter: 1d volume > 1.2x 20-period average (confirms institutional participation)
# - Trend filter: 1w close > 50-period EMA for longs, < 50-period EMA for shorts
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.5x ATR(14) on 12h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines

name = "12h_1d_1w_williamsr_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h Williams %R(14)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.2 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Pre-compute 12h ATR(14) for stoploss
    tr_1 = high_12h - low_12h
    tr_2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    tr[0] = tr_1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R mean reversion OR stoploss hit
            if williams_r[i] > -20 or close_12h[i] < entry_price - 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R mean reversion OR stoploss hit
            if williams_r[i] < -80 or close_12h[i] > entry_price + 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume and trend confirmation
            if vol_spike_aligned[i]:
                # Long: Williams %R crosses above -80 from below (oversold bounce)
                if williams_r[i] > -80 and williams_r[i-1] <= -80 and close_12h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: Williams %R crosses below -20 from above (overbought rejection)
                elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close_12h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals