#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d volume confirmation
# - Primary signal: Williams %R(14) crosses above -80 from below (long) or below -20 from above (short)
# - Volume filter: 1d volume > 1.3x 20-period average volume (institutional participation)
# - Trend filter: price > 50-period EMA on 6h for longs, price < 50-period EMA on 6h for shorts
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 6h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Williams %R identifies overextended moves; volume confirms institutional interest;
#   EMA filter ensures trades align with intermediate-term trend, reducing whipsaw

name = "6h_1d_williamsr_volume_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.3 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Pre-compute 6h EMA(50) for trend filter
    ema_50 = pd.Series(close_6h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Pre-compute 6h ATR(14) for stoploss
    tr_1 = high_6h - low_6h
    tr_2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr_3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    tr[0] = tr_1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (mean reversion) OR stoploss hit
            if williams_r[i] < -50 or close_6h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (mean reversion) OR stoploss hit
            if williams_r[i] > -50 or close_6h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme reversals with volume and trend filters
            if vol_spike_aligned[i]:
                # Long: Williams %R crosses above -80 from below AND price > EMA50
                if williams_r[i] > -80 and williams_r[i-1] <= -80 and close_6h[i] > ema_50[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Williams %R crosses below -20 from above AND price < EMA50
                elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close_6h[i] < ema_50[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals