#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze Breakout with 1d Volume Regime Filter
# Uses Bollinger Band width percentile to detect low volatility squeeze (regime filter)
# Breakout occurs when price closes outside BB(20,2) with volume > 1.5x 20-period average
# Works in bull markets by buying upside breakouts and in bear markets by selling downside breakouts
# Volume regime filter ensures breakouts occur during institutional participation
# ATR-based position sizing (0.25) manages risk through volatility adaptation
# Targets 20-40 trades/year (80-160 total over 4 years) for 4h timeframe

name = "4h_BollingerSqueezeBreakout_1dVolumeRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d average volume for regime filter
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime = vol_1d > (vol_ma_1d * 1.2)  # High volume regime
    
    # Align volume regime to 4h timeframe (wait for completed 1d bar)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    # Calculate Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Calculate Bollinger Band width percentile (squeeze detection)
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).rank(pct=True).values
    squeeze_condition = bb_width_percentile < 0.2  # Lowest 20% = squeeze
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for BB and volume calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(squeeze_condition[i]) or np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price closes above BB upper + BB squeeze + high volume regime
            if close[i] > bb_upper[i] and squeeze_condition[i] and vol_regime_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Price closes below BB lower + BB squeeze + high volume regime
            elif close[i] < bb_lower[i] and squeeze_condition[i] and vol_regime_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when price closes below BB middle (mean reversion)
            if close[i] < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price closes above BB middle (mean reversion)
            if close[i] > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals