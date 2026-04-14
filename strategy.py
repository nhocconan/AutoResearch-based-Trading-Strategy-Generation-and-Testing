#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout with Daily Trend Filter
# Uses Williams fractals (from daily timeframe) to identify potential reversal points.
# Enters long when price breaks above a bullish fractal with daily EMA uptrend.
# Enters short when price breaks below a bearish fractal with daily EMA downtrend.
# Includes volume confirmation to avoid false breakouts.
# Works in bull/bear by trading breakouts in direction of higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams fractals on daily data
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Calculate daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align fractals and EMA to 6s timeframe with proper delay
    # Williams fractals need 2 extra daily bars for confirmation (center + 2 right bars)
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x average volume (50-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=50, min_periods=50).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for EMA calculation and fractal alignment
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Only trade with volume confirmation
        if vol <= 1.5 * avg_vol[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: EMA slope determines trend direction
        if i > start:
            ema_slope = ema_34_aligned[i] - ema_34_aligned[i-1]
        else:
            ema_slope = 0
        
        if position == 0:
            # Long: price breaks above bullish fractal with daily uptrend
            if price > bullish_fractal_aligned[i] and ema_slope > 0:
                position = 1
                signals[i] = position_size
            # Short: price breaks below bearish fractal with daily downtrend
            elif price < bearish_fractal_aligned[i] and ema_slope < 0:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below bearish fractal
            if price < bearish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above bullish fractal
            if price > bullish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Williams_Fractal_Breakout_Trend"
timeframe = "6h"
leverage = 1.0