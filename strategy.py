#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Fractal with 1-day EMA trend filter and volume confirmation.
# Fractals identify potential reversal points; EMA100 confirms long-term trend direction; volume validates strength.
# Long when: bullish fractal forms, price > EMA100, volume > 1.8x 20-period average
# Short when: bearish fractal forms, price < EMA100, volume > 1.8x 20-period average
# Exit when: opposite fractal forms (bearish for long exit, bullish for short exit)
# Works in bull (buy bullish fractals in uptrend) and bear (sell bearish fractals in downtrend).
# Target: 10-20 trades/year per symbol.
name = "6h_WilliamsFractal_EMA100_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA100 for long-term trend
    ema100 = pd.Series(close).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1-day data for Williams Fractals (HTF)
    df_1d = get_htf_data(prices, '1d')
    # Calculate Williams Fractals on 1D data
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align to LTF with 2-bar delay for fractal confirmation (requires 2 future 1D bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for EMA100 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema100[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema = ema100[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bearish = bearish_fractal_aligned[i]
        bullish = bullish_fractal_aligned[i]
        
        if position == 0:
            # Long entry: bullish fractal, price above EMA100, volume spike
            if bullish and price > ema and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish fractal, price below EMA100, volume spike
            elif bearish and price < ema and vol > 1.8 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish fractal forms (potential reversal)
            if bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish fractal forms (potential reversal)
            if bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals