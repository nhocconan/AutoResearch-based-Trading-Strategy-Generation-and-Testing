#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h trend filter and volume confirmation
# - Entry: Williams %R(14) crosses above -80 from below (long) or below -20 from above (short)
# - Trend filter: 12h EMA(50) slope confirms direction (avoid counter-trend trades)
# - Volume: 6h volume > 1.3x 20-period average (institutional participation)
# - Position size: 0.25 discrete levels to minimize fee churn
# - Stoploss: 2.5x ATR(14) on 6h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Williams %R catches reversals; trend filter avoids whipsaws

name = "6h_12h_williamsr_volume_trend_v1"
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
    ema_slope = ema_50 - np.roll(ema_50, 1)  # positive = rising trend
    ema_slope[0] = 0
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.3 * avg_volume_20)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    # Williams %R cross signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]
    williams_long = (williams_r > -80) & (williams_r_prev <= -80)  # cross above -80
    williams_short = (williams_r < -20) & (williams_r_prev >= -20)  # cross below -20
    
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
        if (np.isnan(ema_slope_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 OR stoploss hit
            if williams_r[i] < -50 or close_6h[i] < entry_price - 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 OR stoploss hit
            if williams_r[i] > -50 or close_6h[i] > entry_price + 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R reversals with trend and volume filters
            if vol_spike[i]:
                if williams_long[i] and ema_slope_aligned[i] > 0:  # long in uptrend
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                elif williams_short[i] and ema_slope_aligned[i] < 0:  # short in downtrend
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals