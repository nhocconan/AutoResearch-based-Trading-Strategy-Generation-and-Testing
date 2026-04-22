# 12h Williams Fractal Breakout with Volume Confirmation and Trend Filter
# Hypothesis: Williams Fractal breakouts on 12h chart capture significant support/resistance breaks.
# Long when price breaks above bearish fractal with volume spike and uptrend (price > 12h EMA50).
# Short when price breaks below bullish fractal with volume spike and downtrend (price < 12h EMA50).
# Uses daily Williams Fractals for higher timeframe structure, reducing false signals.
# Designed to work in both bull and bear markets by following breakouts with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Williams Fractals and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Bearish fractal needs 2 extra daily bars for confirmation (after center bar)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    # Bullish fractal needs 2 extra daily bars for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 12h EMA50 for trend filter (using 1d close as proxy for 12h trend)
    # Since we don't have 12h data directly, we use daily close and align
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 20-period average volume for volume spike filter
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        ema50 = ema50_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strong volume spike)
        vol_spike = vol > 2.0 * vol_ma
        
        # Trend filter: price above/below 12h EMA50 (using daily EMA as proxy)
        uptrend = price > ema50
        downtrend = price < ema50
        
        if position == 0:
            # Long: price breaks above bearish fractal (resistance) with volume spike and uptrend
            if price > bear_fractal and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal (support) with volume spike and downtrend
            elif price < bull_fractal and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite fractal or volume/spike conditions fail
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price drops back below bullish fractal (support) or volume spike fails
                if price < bull_fractal or not vol_spike:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price rises back above bearish fractal (resistance) or volume spike fails
                if price > bear_fractal or not vol_spike:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0