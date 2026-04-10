#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1w EMA trend filter + volume confirmation
# - Primary signal: Williams %R(14) on 12h crosses above -50 (bullish) or below -50 (bearish)
# - Trend filter: 1w EMA(50) slope > 0 for longs, < 0 for shorts (institutional trend)
# - Volume filter: 12h volume > 1.3x 20-period average volume (momentum confirmation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 12h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Williams %R captures momentum shifts; EMA filter avoids counter-trend trades

name = "12h_1w_williamsr_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) and its slope for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_slope = ema_50 - np.roll(ema_50, 1)
    ema_slope[0] = 0
    ema_slope_pos = ema_slope > 0  # Uptrend
    ema_slope_neg = ema_slope < 0  # Downtrend
    ema_slope_pos_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_pos)
    ema_slope_neg_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_neg)
    
    # Pre-compute 12h Williams %R(14)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Avoid division by zero
    
    # Pre-compute 12h volume spike filter
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.3 * avg_volume_20)
    
    # Pre-compute 12h ATR(14) for stoploss
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_slope_pos_aligned[i]) or
            np.isnan(ema_slope_neg_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 OR stoploss hit
            if williams_r[i] < -50 or close_12h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 OR stoploss hit
            if williams_r[i] > -50 or close_12h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R cross with trend and volume filters
            if vol_spike[i]:
                # Long: Williams %R crosses above -50 in uptrend
                if williams_r[i] > -50 and williams_r[i-1] <= -50 and ema_slope_pos_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: Williams %R crosses below -50 in downtrend
                elif williams_r[i] < -50 and williams_r[i-1] >= -50 and ema_slope_neg_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals