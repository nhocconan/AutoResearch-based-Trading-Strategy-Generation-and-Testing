#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation
# - Long when price breaks above the most recent bearish Williams fractal (swing high) with volume > 1.5x 20-bar average AND 1d close > 1d EMA50
# - Short when price breaks below the most recent bullish Williams fractal (swing low) with volume > 1.5x 20-bar average AND 1d close < 1d EMA50
# - Exit when price retraces to the 50% level of the last swing range OR volume drops below 0.7x average
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Williams fractals require 2-bar confirmation after the center bar (additional_delay_bars=2)
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Focus on BTC/ETH; SOL-only strategies are low value and will be discarded

name = "6h_1d_williams_fractal_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    swing_high = 0.0
    swing_low = 0.0
    in_swing = False
    
    # Pre-compute aligned 1d data properly
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Align them to 6h timeframe
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Compute Williams fractals on 1d data
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or np.isnan(c_1d_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Update swing points from Williams fractals
        # Bearish fractal (swing high) - value is the high price at the fractal point
        if bearish_fractal_aligned[i] > 0:
            swing_high = bearish_fractal_aligned[i]
            in_swing = True
        # Bullish fractal (swing low) - value is the low price at the fractal point
        if bullish_fractal_aligned[i] > 0:
            swing_low = bullish_fractal_aligned[i]
            in_swing = True
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > most recent bearish fractal (swing high) with volume spike AND 1d uptrend
            if (swing_high > 0 and 
                prices['close'].iloc[i] > swing_high and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                signals[i] = 0.25
            # Short breakdown: price < most recent bullish fractal (swing low) with volume spike AND 1d downtrend
            elif (swing_low > 0 and 
                  prices['close'].iloc[i] < swing_low and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Calculate 50% retracement level of the last swing range
            if swing_high > 0 and swing_low > 0:
                swing_range = swing_high - swing_low
                if swing_range > 0:
                    if position == 1:  # Long position
                        retracement_level = swing_low + (swing_range * 0.5)
                        # Exit conditions:
                        # 1. Price retraces to 50% of swing range
                        # 2. Volume drops below 0.7x average (loss of momentum)
                        if (prices['close'].iloc[i] <= retracement_level or 
                            vol_weak.iloc[i]):
                            position = 0
                            entry_price = 0.0
                            signals[i] = 0.0
                        else:
                            signals[i] = 0.25  # Hold long
                    elif position == -1:  # Short position
                        retracement_level = swing_high - (swing_range * 0.5)
                        # Exit conditions:
                        # 1. Price retraces to 50% of swing range (from high)
                        # 2. Volume drops below 0.7x average (loss of momentum)
                        if (prices['close'].iloc[i] >= retracement_level or 
                            vol_weak.iloc[i]):
                            position = 0
                            entry_price = 0.0
                            signals[i] = 0.0
                        else:
                            signals[i] = -0.25  # Hold short
                else:
                    # Invalid range, hold position
                    if position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
            else:
                # No valid swing yet, hold position
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals