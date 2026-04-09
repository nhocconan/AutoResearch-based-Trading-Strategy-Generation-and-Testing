#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Uses 1d Camarilla pivot levels (H4/L4) as key support/resistance for breakouts
# - Trend filter: 1d EMA(50) > EMA(200) for long bias, EMA(50) < EMA(200) for short bias
# - Volume confirmation: 4h volume > 2.0x 20-period average to ensure breakout strength
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Novelty: Camarilla pivots from 1d provide institutional levels; EMA filter ensures trading with higher timeframe trend
# - Works in both bull/bear: Pivots work in ranging markets, EMA filter adapts to trend direction

name = "4h_1d_camarilla_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) and EMA(200) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50 = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = close_1d_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200  # Long bias when EMA50 > EMA200
    ema_bearish = ema_50 < ema_200  # Short bias when EMA50 < EMA200
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: H4 = Close + 1.1*(High-Low)*1.1/2, L4 = Close - 1.1*(High-Low)*1.1/2
    # Simplified: H4 = Close + 1.1*(High-Low), L4 = Close - 1.1*(High-Low)
    camarilla_high = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_low = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align 1d indicators to 4h timeframe (completed 1d bar only)
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h volume > 2.0x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            np.isnan(ema_bullish_aligned[i]) or
            np.isnan(ema_bearish_aligned[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and trend filter
            # Long: price breaks above Camarilla H4 AND volume spike AND bullish 1d trend
            if high[i] >= camarilla_high_aligned[i] and volume_spike[i] and ema_bullish_aligned[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price breaks below Camarilla L4 AND volume spike AND bearish 1d trend
            elif low[i] <= camarilla_low_aligned[i] and volume_spike[i] and ema_bearish_aligned[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals