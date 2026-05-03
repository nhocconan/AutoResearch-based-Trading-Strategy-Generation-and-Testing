#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + RSI(2) Extreme Reversion
# Uses Bollinger Band Width percentile to detect regime: 
# - High BBW (>80th percentile) = expansion/trending -> avoid entries
# - Low BBW (<20th percentile) = contraction/squeeze -> mean reversion ripe
# In squeeze regime: RSI(2) < 10 = long, RSI(2) > 90 = short
# Volume confirmation: current volume > 1.5x 20-period EMA to filter false signals
# Works in both bull/bear markets as squeeze/mean reversion is regime-independent
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

name = "6h_BBW_Regime_RSI2_Extreme"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20, 2) on 6h
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean().values
    dev = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = basis + (2 * dev)
    lower_band = basis - (2 * dev)
    bb_width = (upper_band - lower_band) / basis  # Normalized width
    
    # Calculate BBW percentile rank (50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate RSI(2) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(rsi_2[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Regime: Low BBW (<20th percentile) = squeeze/mean reversion ripe
        low_bbw_regime = bb_width_percentile[i] < 20
        
        # Mean reversion signals in squeeze regime
        if position == 0:
            if low_bbw_regime and volume_spike:
                if rsi_2[i] < 10:  # Extreme oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi_2[i] > 90:  # Extreme overbought
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: RSI(2) > 50 (mean reversion complete) OR BBW expansion (>80)
            if rsi_2[i] > 50 or bb_width_percentile[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI(2) < 50 (mean reversion complete) OR BBW expansion (>80)
            if rsi_2[i] < 50 or bb_width_percentile[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals