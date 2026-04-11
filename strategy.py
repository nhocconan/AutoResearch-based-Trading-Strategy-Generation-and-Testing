#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ma_volume_squeeze_v1"
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
    
    # Daily EMA50 for trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h MA20 for dynamic support/resistance
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Bollinger Bands for squeeze detection (20, 2)
    ma_20_bb = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20_bb + 2 * std_20
    lower_bb = ma_20_bb - 2 * std_20
    
    # Volume spike detection (volume > 2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily EMA to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ma_20[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        ma_20_val = ma_20[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # Bollinger Band squeeze: bandwidth < 5% of price (low volatility)
        bb_width = (upper_bb[i] - lower_bb[i]) / ma_20_bb[i] if ma_20_bb[i] != 0 else 1
        squeeze = bb_width < 0.05
        
        # Volume confirmation: volume spike
        volume_spike = volume_current > 2.0 * vol_ma_20[i]
        
        # Trend filter: price relative to daily EMA50
        above_trend = price_close > ema_trend
        below_trend = price_close < ema_trend
        
        # Mean reversion signals from Bollinger Bands with volume confirmation
        long_signal = False
        short_signal = False
        
        # Long: price at lower BB during squeeze + volume spike + above daily EMA trend
        if squeeze and volume_spike and price_close <= lower_bb[i] and above_trend:
            long_signal = True
        
        # Short: price at upper BB during squeeze + volume spike + below daily EMA trend
        if squeeze and volume_spike and price_close >= upper_bb[i] and below_trend:
            short_signal = True
        
        # Exit: return to middle Bollinger Band (mean reversion complete)
        exit_long = price_close >= ma_20_bb[i]
        exit_short = price_close <= ma_20_bb[i]
        
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

# Hypothesis: 12h Bollinger Band squeeze strategy with daily EMA50 trend filter and volume confirmation.
# Enters long when price touches lower BB during low volatility squeeze with volume spike and price above daily EMA50.
# Enters short when price touches upper BB during squeeze with volume spike and price below daily EMA50.
# Exits when price returns to middle Bollinger Band (20-period MA).
# Uses Bollinger Band squeeze (<5% width) to identify low volatility periods primed for expansion.
# Volume confirmation (>2x 20-period average) ensures institutional participation.
# Daily EMA50 filter ensures trades align with higher timeframe trend.
# Target: 15-25 trades per year to minimize fee drag while capturing explosive moves after consolidation.
# Works in both bull and bear markets by trading mean reversion explosions from squeezed conditions.