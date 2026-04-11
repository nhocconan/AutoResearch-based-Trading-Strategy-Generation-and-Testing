#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with volume confirmation and 1w trend filter
# - Enter long when price breaks above H4 resistance with volume > 2.0x 20-period volume SMA AND 1w close > 1w EMA20
# - Enter short when price breaks below L4 support with volume > 2.0x 20-period volume SMA AND 1w close < 1w EMA20
# - Exit: price retreats to H3 (for longs) or L3 (for shorts)
# - Breakout logic captures momentum in both bull and bear markets
# - Volume confirmation ensures institutional participation
# - 1w EMA20 filter avoids counter-trend trades in strong weekly trends
# - Target: 10-25 trades/year to minimize fee drag while capturing strong moves

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla calculation and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute Camarilla levels for 1d data (based on previous day's OHLC)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_range = high_1d - low_1d
    h4 = close_1d + 1.5 * camarilla_range
    h3 = close_1d + 1.125 * camarilla_range
    l3 = close_1d - 1.125 * camarilla_range
    l4 = close_1d - 1.5 * camarilla_range
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute EMA20 for 1w close (trend filter)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute 1w close aligned for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    for i in range(20, n):  # Start after 20-bar warmup for volume SMA
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        volume_confirm = volume[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1w close vs EMA20
        uptrend = close_1w_aligned[i] > ema_20_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Camarilla breakout signals (using 1d close for breakout confirmation)
        breakout_high = close[i] > h4_aligned[i]  # Price closed above H4 resistance
        breakout_low = close[i] < l4_aligned[i]   # Price closed below L4 support
        
        # Exit conditions: price retreats to H3/L3 levels
        retreat_high = close[i] < h3_aligned[i]  # Price closed below H3 (exit long)
        retreat_low = close[i] > l3_aligned[i]   # Price closed above L3 (exit short)
        
        # Trading logic
        if volume_confirm:
            # Long: H4 breakout in uptrend
            if breakout_high and uptrend:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: L4 breakout in downtrend
            elif breakout_low and downtrend:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and retreat_high:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and retreat_low:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals