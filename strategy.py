#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1-day trend filter and volume confirmation
# Uses Williams Alligator (Jaw: SMA13, Teeth: SMA8, Lips: SMA5) on 12h for trend direction
# Long when Lips > Teeth > Jaw and price above Lips; Short when Lips < Teeth < Jaw and price below Lips
# Daily close > daily EMA50 as trend filter (only long in daily uptrend, short in daily downtrend)
# Volume confirmation > 1.3x 20-period EMA on 12h to reduce false signals
# Designed for ~15-25 trades/year with clear trend-following logic
# Works in bull markets via bullish alignment + uptrend and in bear markets via bearish alignment + downtrend
# Position size: 0.25 to balance return and drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator for 12h (SMA5, SMA8, SMA13)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values  # SMA5
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values  # SMA8
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # SMA13
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume moving average for confirmation (20-period EMA on 12h)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned daily EMA50
        ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)[i]
        
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or np.isnan(ema50_1d_aligned) or np.isnan(vol_ma[i]):
            continue
        
        # Daily trend filter: only long in uptrend, only short in downtrend
        # Need to get the last known daily close for comparison
        # Find the index of the last completed daily bar up to current 12h bar
        # Since we can't easily compute this without datetime math, we'll use a simplification:
        # Use the daily EMA50 value as proxy - if price > EMA50, consider uptrend
        # This is not perfect but avoids look-ahead and uses available data
        price_vs_ema = close[i] > ema50_1d_aligned  # Simple price vs EMA comparison
        
        # Williams Alligator signals
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])  # Lips > Teeth > Jaw
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])  # Lips < Teeth < Jaw
        price_above_lips = close[i] > lips[i]
        price_below_lips = close[i] < lips[i]
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Long signal: bullish alignment + price above lips + daily uptrend + volume
        if position == 0 and bullish_alignment and price_above_lips and price_vs_ema and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short signal: bearish alignment + price below lips + daily downtrend + volume
        elif position == 0 and bearish_alignment and price_below_lips and not price_vs_ema and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: loss of alignment or price crosses lips in opposite direction
        elif position != 0:
            if position == 1 and (not bullish_alignment or close[i] < lips[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (not bearish_alignment or close[i] > lips[i]):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1dTrend_Filter_Volume"
timeframe = "12h"
leverage = 1.0