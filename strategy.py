#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA(50) trend filter and volume confirmation
# - Uses 4h EMA(50) for trend direction (avoid counter-trend trades)
# - Uses 1h Camarilla pivot levels (H3/L3) for precise entry timing
# - Volume confirmation: volume > 1.8x 20-period average to ensure breakout strength
# - Session filter: 08-20 UTC to avoid low-liquidity Asian session noise
# - ATR(14) trailing stop at 2.5x ATR from extreme for risk control
# - Position size: 0.20 (20% of capital) - discrete level to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h timeframe guidelines
# - Novelty: Combining proven Camarilla pivot structure with 4h trend filter for 1h precision entries

name = "1h_4h_camarilla_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h Camarilla pivot levels (based on previous 4h bar's OHLC)
    # Calculate Camarilla levels from 4h OHLC (aligned to 1h)
    cam_h3 = align_htf_to_ltf(prices, df_4h, df_4h['close'].values + 1.1 * (df_4h['high'].values - df_4h['low'].values))
    cam_l3 = align_htf_to_ltf(prices, df_4h, df_4h['close'].values - 1.1 * (df_4h['high'].values - df_4h['low'].values))
    
    # 1h volume > 1.8x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    # 1h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(cam_h3[i]) or
            np.isnan(cam_l3[i]) or
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
            
            # Exit conditions: price retraces 2.5x ATR from high
            if low[i] <= highest_since_entry - (2.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low
            if high[i] >= lowest_since_entry + (2.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and 4h EMA trend filter
            # Long: price breaks above Camarilla H3 AND price > 4h EMA50 AND volume spike
            if (high[i] >= cam_h3[i] and 
                close[i] > ema_50_4h_aligned[i] and
                volume_spike[i]):
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.20
            # Short: price breaks below Camarilla L3 AND price < 4h EMA50 AND volume spike
            elif (low[i] <= cam_l3[i] and 
                  close[i] < ema_50_4h_aligned[i] and
                  volume_spike[i]):
                position = -1
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.20
    
    return signals