#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Bollinger Band width < 20th percentile indicates low volatility squeeze.
# Breakout above upper band = long, below lower band = short.
# Requires 1d EMA50 trend alignment and volume spike (>2x 20-period MA).
# Designed for 12-37 trades/year to minimize fee drag and work in both bull/bear markets via volatility expansion trades.

name = "6h_BollingerSqueeze_Breakout_1dEMA50_Trend_VolumeConfirmation"
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
    
    # Get 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2) on 6h data
    close_s = pd.Series(close)
    basis = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    dev = 2.0 * close_s.ewm(span=20, adjust=False, min_periods=20).std().values
    upper = basis + dev
    lower = basis - dev
    
    # Calculate Bollinger Band Width percentile (20-period lookback)
    bbw = (upper - lower) / basis
    bbw_s = pd.Series(bbw)
    bbw_percentile = bbw_s.rolling(window=20, min_periods=20).rank(pct=True).values
    
    # Squeeze condition: BBW < 20th percentile
    squeeze = bbw_percentile < 0.20
    
    # Get 1d EMA50 trend filter - ONCE before loop
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for squeeze breakout with volume confirmation and trend alignment
            if squeeze[i-1] and not squeeze[i]:  # squeeze just released
                # Long breakout: price closes above upper band
                if close[i] > upper[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price closes below lower band
                elif close[i] < lower[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to basis OR volatility expands too much (breakout fails)
            if close[i] < basis[i] or bbw_percentile[i] > 0.80:  # BBW > 80th percentile = high volatility
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to basis OR volatility expands too much
            if close[i] > basis[i] or bbw_percentile[i] > 0.80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals