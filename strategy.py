#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level from prior 1d in uptrend (close > EMA50)
# - Short when price breaks below Camarilla L3 level from prior 1d in downtrend (close < EMA50)
# - Volume filter: 12h volume > 1.5x 20-period average to confirm breakout
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 12h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Breakouts capture strong moves; trend filter avoids counter-trend trades

name = "12h_1d_camarilla_breakout_v1"
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
    
    # Pre-compute 12h Camarilla levels from prior 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    H3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    L3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (use prior 1d levels)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Pre-compute 12h volume spike filter
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.5 * avg_volume_20)
    
    # Pre-compute 12h ATR(14) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
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
        if (np.isnan(ema_50_aligned[i]) or np.isnan(H3_aligned[i]) or
            np.isnan(L3_aligned[i]) or np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below L3 (mean reversion) OR stoploss hit
            if close_12h[i] < L3_aligned[i] or close_12h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above H3 (mean reversion) OR stoploss hit
            if close_12h[i] > H3_aligned[i] or close_12h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i]:
                # Long: price breaks above H3 in uptrend (close > EMA50)
                if close_12h[i] > H3_aligned[i] and close_12h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: price breaks below L3 in downtrend (close < EMA50)
                elif close_12h[i] < L3_aligned[i] and close_12h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals