#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Volume Regime Filter
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets.
# Breakout direction confirmed by 1d volume regime (high volume = institutional participation).
# Designed for 12-37 trades/year on 6h to minimize fee drag while capturing volatility expansion.
# Works in bull markets via upside breakouts and in bear markets via downside breakdowns.

name = "6h_BollingerSqueeze_Breakout_1dVolumeRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume regime: 20-period volume EMA
    vol_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_high_regime = volume_1d > (vol_ema_20_1d * 1.5)  # High volume regime
    
    # Align 1d volume regime to 6h timeframe (wait for completed 1d bar)
    volume_high_aligned = align_htf_to_ltf(prices, df_1d, volume_high_regime)
    
    # Calculate 6h Bollinger Bands (20, 2.0)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + (2.0 * dev)
    lower_band = basis - (2.0 * dev)
    
    # Bollinger Band Width (BBW) for squeeze detection
    bbw = (upper_band - lower_band) / basis
    
    # Squeeze condition: BBW below 20-period BBW EMA (low volatility)
    bbw_ema_20 = pd.Series(bbw).ewm(span=20, adjust=False, min_periods=20).mean().values
    squeeze = bbw < bbw_ema_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(bbw[i]) or np.isnan(squeeze[i]) or 
            np.isnan(volume_high_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: squeeze breakout above upper band AND volume high regime
            if squeeze[i-1] and close[i] > upper_band[i] and volume_high_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze breakout below lower band AND volume high regime
            elif squeeze[i-1] and close[i] < lower_band[i] and volume_high_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below basis (mean reversion) OR squeeze fires in opposite direction
            if close[i] < basis[i] or (squeeze[i] and close[i] < lower_band[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above basis (mean reversion) OR squeeze fires in opposite direction
            if close[i] > basis[i] or (squeeze[i] and close[i] > upper_band[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals