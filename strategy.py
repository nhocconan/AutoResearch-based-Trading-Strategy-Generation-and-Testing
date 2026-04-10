#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d trend filter + volume confirmation
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels from prior 1d
# - Trend filter: 1d close > EMA(50) for longs, < EMA(50) for shorts (institutional trend)
# - Volume filter: 12h volume > 1.5x 20-period average volume (momentum confirmation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 12h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Camarilla levels provide structure in ranging markets; trend filter avoids counter-trend trades

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
    
    # Pre-compute 12h volume spike filter
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla H3 OR stoploss hit
            if i >= 1:
                # Get prior 1d high/low/close for Camarilla calculation
                idx_1d = i // 16  # 16x 12h bars per 1d
                if idx_1d < len(df_1d):
                    h1 = df_1d['high'].iloc[idx_1d-1] if idx_1d-1 >= 0 else df_1d['high'].iloc[0]
                    l1 = df_1d['low'].iloc[idx_1d-1] if idx_1d-1 >= 0 else df_1d['low'].iloc[0]
                    c1 = df_1d['close'].iloc[idx_1d-1] if idx_1d-1 >= 0 else df_1d['close'].iloc[0]
                    
                    # Camarilla levels
                    camarilla_h3 = c1 + (h1 - l1) * 1.1 / 4
                    camarilla_l3 = c1 - (h1 - l1) * 1.1 / 4
                    
                    if close_12h[i] < camarilla_h3 or close_12h[i] < entry_price - 2.0 * atr_14[i]:
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = 0.25
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla L3 OR stoploss hit
            if i >= 1:
                # Get prior 1d high/low/close for Camarilla calculation
                idx_1d = i // 16  # 16x 12h bars per 1d
                if idx_1d < len(df_1d):
                    h1 = df_1d['high'].iloc[idx_1d-1] if idx_1d-1 >= 0 else df_1d['high'].iloc[0]
                    l1 = df_1d['low'].iloc[idx_1d-1] if idx_1d-1 >= 0 else df_1d['low'].iloc[0]
                    c1 = df_1d['close'].iloc[idx_1d-1] if idx_1d-1 >= 0 else df_1d['close'].iloc[0]
                    
                    # Camarilla levels
                    camarilla_h3 = c1 + (h1 - l1) * 1.1 / 4
                    camarilla_l3 = c1 - (h1 - l1) * 1.1 / 4
                    
                    if close_12h[i] > camarilla_l3 or close_12h[i] > entry_price + 2.0 * atr_14[i]:
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i] and i >= 1:
                # Get prior 1d high/low/close for Camarilla calculation
                idx_1d = i // 16  # 16x 12h bars per 1d
                if idx_1d < len(df_1d):
                    h1 = df_1d['high'].iloc[idx_1d-1] if idx_1d-1 >= 0 else df_1d['high'].iloc[0]
                    l1 = df_1d['low'].iloc[idx_1d-1] if idx_1d-1 >= 0 else df_1d['low'].iloc[0]
                    c1 = df_1d['close'].iloc[idx_1d-1] if idx_1d-1 >= 0 else df_1d['close'].iloc[0]
                    
                    # Camarilla levels
                    camarilla_h3 = c1 + (h1 - l1) * 1.1 / 4
                    camarilla_l3 = c1 - (h1 - l1) * 1.1 / 4
                    
                    # Long: price breaks above H3 in uptrend (close > EMA50)
                    if close_12h[i] > camarilla_h3 and close_12h[i] > ema_50_aligned[i]:
                        position = 1
                        entry_price = close_12h[i]
                        signals[i] = 0.25
                    # Short: price breaks below L3 in downtrend (close < EMA50)
                    elif close_12h[i] < camarilla_l3 and close_12h[i] < ema_50_aligned[i]:
                        position = -1
                        entry_price = close_12h[i]
                        signals[i] = -0.25
    
    return signals