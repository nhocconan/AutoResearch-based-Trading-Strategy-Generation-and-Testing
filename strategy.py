#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h trend filter and volume confirmation
# - Primary signal: 6h price breaks above R4 or below S4 Camarilla levels from prior 12h bar
# - Trend filter: 12h EMA(20) slope > 0 for longs, < 0 for shorts (avoid counter-trend)
# - Volume filter: 6h volume > 1.5x 24-period average volume (momentum confirmation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(10) on 6h for tight risk control
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Breakouts capture momentum; EMA filter avoids false breakouts in ranging markets

name = "6h_12h_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(20) and its slope for trend filter
    close_12h = df_12h['close'].values
    ema_20 = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_slope = ema_20 - np.roll(ema_20, 1)
    ema_slope[0] = 0
    ema_slope_pos = ema_slope > 0  # Uptrend
    ema_slope_neg = ema_slope < 0  # Downtrend
    ema_slope_pos_aligned = align_htf_to_ltf(prices, df_12h, ema_slope_pos)
    ema_slope_neg_aligned = align_htf_to_ltf(prices, df_12h, ema_slope_neg)
    
    # Pre-compute 12h Camarilla pivot levels (based on prior 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r4 = pivot + range_12h * 1.500  # R4
    s4 = pivot - range_12h * 1.500  # S4
    r3 = pivot + range_12h * 1.250  # R3
    s3 = pivot - range_12h * 1.250  # S3
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_24 = pd.Series(volume_6h).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_24)
    
    # Pre-compute 6h ATR(10) for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_slope_pos_aligned[i]) or np.isnan(ema_slope_neg_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_10[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price drops below R3 OR stoploss hit
            if close_6h[i] < r3_aligned[i] or close_6h[i] < entry_price - 1.5 * atr_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above S3 OR stoploss hit
            if close_6h[i] > s3_aligned[i] or close_6h[i] > entry_price + 1.5 * atr_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i]:
                # Long: price breaks above R4 in uptrend
                if close_6h[i] > r4_aligned[i] and ema_slope_pos_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: price breaks below S4 in downtrend
                elif close_6h[i] < s4_aligned[i] and ema_slope_neg_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals