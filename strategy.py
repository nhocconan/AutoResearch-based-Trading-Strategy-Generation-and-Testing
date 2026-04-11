#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_vwap_trend_squeeze_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Daily VWAP for trend direction (more robust than EMA)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # 12h VWAP for dynamic support/resistance
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    vwap = vwap.values
    
    # Bollinger Bands for squeeze detection (20, 2) on 12h
    vwap_series = pd.Series(vwap)
    vwap_ma_20 = vwap_series.rolling(window=20, min_periods=20).mean().values
    vwap_std_20 = vwap_series.rolling(window=20, min_periods=20).std().values
    upper_band = vwap_ma_20 + 2 * vwap_std_20
    lower_band = vwap_ma_20 - 2 * vwap_std_20
    
    # Volume spike detection (volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily VWAP to 12h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(vwap_ma_20[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vwap_val = vwap[i]
        vwap_1d_val = vwap_1d_aligned[i]
        
        # Bollinger Band squeeze: bandwidth < 4% of VWAP (low volatility)
        bb_width = (upper_band[i] - lower_band[i]) / vwap_ma_20[i] if vwap_ma_20[i] != 0 else 1
        squeeze = bb_width < 0.04
        
        # Volume confirmation: volume spike
        volume_spike = volume_current > 1.8 * vol_ma_20[i]
        
        # Trend filter: price relative to daily VWAP
        above_trend = price_close > vwap_1d_val
        below_trend = price_close < vwap_1d_val
        
        # Mean reversion signals from Bollinger Bands with volume confirmation
        long_signal = False
        short_signal = False
        
        # Long: price at lower BB during squeeze + volume spike + above daily VWAP trend
        if squeeze and volume_spike and price_close <= lower_band[i] and above_trend:
            long_signal = True
        
        # Short: price at upper BB during squeeze + volume spike + below daily VWAP trend
        if squeeze and volume_spike and price_close >= upper_band[i] and below_trend:
            short_signal = True
        
        # Exit: return to VWAP (mean reversion complete)
        exit_long = price_close >= vwap_val
        exit_short = price_close <= vwap_val
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h VWAP Bollinger Band squeeze strategy with daily VWAP trend filter and volume confirmation.
# Enters long when price touches lower Bollinger Band during low volatility squeeze with volume spike and price above daily VWAP.
# Enters short when price touches upper Bollinger Band during squeeze with volume spike and price below daily VWAP.
# Exits when price returns to 12h VWAP (mean reversion complete).
# Uses Bollinger Band squeeze (<4% width) to identify low volatility periods primed for expansion.
# Volume confirmation (>1.8x 20-period average) ensures institutional participation.
# Daily VWAP filter ensures trades align with higher timeframe trend.
# Target: 15-25 trades per year to minimize fee drag while capturing explosive moves after consolidation.
# Works in both bull and bear markets by trading mean reversion explosions from squeezed conditions.
# VWAP is more robust than EMA for trending markets and adapts to volume profile.