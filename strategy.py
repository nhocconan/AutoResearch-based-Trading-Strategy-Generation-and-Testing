#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
# - Uses 4h Camarilla pivot levels (H3/L3) for directional bias
# - 1h entry: break of Camarilla level with volume spike (>2x 20-period avg)
# - 1d trend filter: only long when price > 50 EMA, short when price < 50 EMA
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.20 (20% of capital) - discrete level to minimize fee churn
# - Session filter: 08-20 UTC to avoid low-volume Asian session
# - Target: 60-150 total trades over 4 years = 15-37/year for 1h
# - Novelty: Combines HTF Camarilla structure with LTF timing and trend alignment
# - Works in bull/bear: Camarilla adapts to volatility, trend filter avoids counter-trend traps

name = "1h_4h_1d_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivot levels (H3, L3)
    camarilla_h3_4h = close_4h + (1.1 * (high_4h - low_4h) / 4)
    camarilla_l3_4h = close_4h - (1.1 * (high_4h - low_4h) / 4)
    
    # Align Camarilla levels to 1h timeframe (completed 4h bar only)
    camarilla_h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h volume > 2.0x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # 1h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3_4h_aligned[i]) or 
            np.isnan(camarilla_l3_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0 or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR Camarilla L3 touch
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or \
               low[i] <= camarilla_l3_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR Camarilla H3 touch
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or \
               high[i] >= camarilla_h3_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation AND trend alignment
            # Long: price breaks above Camarilla H3 AND volume spike AND price > 1d EMA50
            if (high[i] >= camarilla_h3_4h_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.20
            # Short: price breaks below Camarilla L3 AND volume spike AND price < 1d EMA50
            elif (low[i] <= camarilla_l3_4h_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.20
    
    return signals