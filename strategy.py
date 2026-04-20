#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Bollinger Band Squeeze Breakout with Volume Confirmation
# - Uses 1w Bollinger Bands (20, 2) to detect volatility squeeze (bandwidth < 20th percentile)
# - Long when price breaks above upper band + volume > 1.5x 20-period average
# - Short when price breaks below lower band + volume > 1.5x 20-period average
# - Exit when price returns to middle band (mean reversion) or opposite band break
# - Designed for 1d timeframe with low-frequency, high-conviction trades
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)
# - Works in bull/bear markets: squeeze breakouts capture volatility expansion regimes

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Bollinger Band calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands on weekly timeframe
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    middle_band = sma_20
    
    # Calculate Bollinger Band Width for squeeze detection
    bb_width = (upper_band - lower_band) / middle_band
    
    # Calculate 20th percentile of BB width for squeeze threshold (using expanding window)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct20 = bb_width_series.expanding(min_periods=20).quantile(0.20).values
    
    # Squeeze condition: BB width < 20th percentile (low volatility)
    squeeze = bb_width < bb_width_pct20
    
    # Align weekly indicators to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    middle_band_aligned = align_htf_to_ltf(prices, df_1w, middle_band)
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze.astype(float))
    
    # Daily volume confirmation
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Bollinger Band warmup
        # Skip if NaN in indicators
        if np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or \
           np.isnan(middle_band_aligned[i]) or np.isnan(squeeze_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        
        if position == 0:
            # Look for volatility squeeze breakout with volume confirmation
            if squeeze_aligned[i] > 0.5:  # In squeeze condition
                # Long breakout: price above upper band + volume confirmation
                if price > upper_band_aligned[i] and vol > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below lower band + volume confirmation
                elif price < lower_band_aligned[i] and vol > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band or breaks below lower band
            if price < middle_band_aligned[i] or price < lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band or breaks above upper band
            if price > middle_band_aligned[i] or price > upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBB_SqueezeBreakout_Volume"
timeframe = "1d"
leverage = 1.0