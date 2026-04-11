#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume spike and 1d trend filter
# - Enter long when price breaks above Camarilla H3 resistance AND 12h volume > 2.0x 20-period volume SMA AND 1d close > 1d EMA50
# - Enter short when price breaks below Camarilla L3 support AND 12h volume > 2.0x 20-period volume SMA AND 1d close < 1d EMA50
# - Exit: price moves to Camarilla H4 (for longs) or L4 (for shorts) or opposite pivot touch
# - Camarilla pivots provide mathematical support/resistance levels
# - Volume confirmation ensures institutional participation
# - 1d EMA50 filter ensures alignment with daily trend
# - Target: 20-50 trades/year to minimize fee drag while capturing high-probability breakouts

name = "4h_12h_1d_camarilla_breakout_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for trend filter and Camarilla calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 12h data ONCE before loop for volume confirmation (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
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
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Pre-compute EMA50 for 1d close (trend filter)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d close aligned for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Pre-compute volume SMA for 12h data (20-period)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute 12h volume aligned for volume confirmation
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    for i in range(50, n):  # Start after 50-bar warmup for EMA50
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or
            np.isnan(volume_sma_20_12h_aligned[i]) or np.isnan(volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 2.0x 20-period volume SMA
        vol_confirm = volume_12h_aligned[i] > 2.0 * volume_sma_20_12h_aligned[i]
        
        # Trend filter: 1d close vs EMA50
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Camarilla pivot breakout signals (using 4h close for breakout detection)
        breakout_h3 = close[i] > h3_aligned[i]  # Price closed above H3 level
        breakout_l3 = close[i] < l3_aligned[i]  # Price closed below L3 level
        touch_h4 = high[i] >= h4_aligned[i] and low[i] <= h4_aligned[i]  # Price touched H4 level (exit for longs)
        touch_l4 = high[i] >= l4_aligned[i] and low[i] <= l4_aligned[i]  # Price touched L4 level (exit for shorts)
        touch_opposite_long = touch_l3  # Exit long when price touches opposite L3
        touch_opposite_short = touch_h3  # Exit short when price touches opposite H3
        
        # Exit conditions
        exit_long = touch_h4 or touch_opposite_long  # Exit long when price reaches H4 or touches L3
        exit_short = touch_l4 or touch_opposite_short  # Exit short when price reaches L4 or touches H3
        
        # Trading logic
        if vol_confirm:
            # Long: H3 breakout in uptrend
            if breakout_h3 and uptrend:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.30
                else:
                    signals[i] = 0.30
            # Short: L3 breakout in downtrend
            elif breakout_l3 and downtrend:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.30
                else:
                    signals[i] = -0.30
            else:
                # Check for exits
                if position == 1 and exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals