#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: TRIX (triple exponential average) identifies momentum with less lag than MACD. Combined with volume spike (>2x 20-period average) and choppiness regime filter (CHOP > 61.8 for ranging markets), this strategy captures mean-reversion bursts in ranging conditions and momentum continuation in trending markets. Uses 4h primary timeframe with 12h HTF for trend context. Designed for low trade frequency (target 20-40 trades/year) to minimize fee drag while maintaining edge in both bull and bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate TRIX on close (primary indicator)
    # TRIX = EMA(EMA(EMA(close, period), period), period) - 1 period ago
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # Percentage change
    trix_values = trix.values
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix_values)
    
    # Calculate 12h EMA50 for trend context
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness Index regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # First bar has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high - min_low
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # Avoid division by zero
    chop = 100 * np.log10(atr_sum / chop_denom) / np.log10(14)
    
    # Regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    chop_regime_ranging = chop > 61.8
    chop_regime_trending = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need TRIX (12*3=34), EMA50 (50), volume avg (20), CHOP (14)
    start_idx = max(50, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        trix_val = trix_aligned[i]
        ema_12h_val = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        is_ranging = chop_regime_ranging[i]
        is_trending = chop_regime_trending[i]
        
        if position == 0:
            # In ranging market: mean reversion on TRIX extremes
            if is_ranging:
                # Long when TRIX is deeply oversold and volume spikes
                if trix_val < -0.5 and vol_spike:
                    signals[i] = size
                    position = 1
                # Short when TRIX is deeply overbought and volume spikes
                elif trix_val > 0.5 and vol_spike:
                    signals[i] = -size
                    position = -1
            # In trending market: momentum continuation with volume
            elif is_trending:
                # Long when TRIX turns up above zero with volume spike and price above EMA
                if trix_val > 0 and trix_val > trix_aligned[i-1] and vol_spike and close[i] > ema_12h_val:
                    signals[i] = size
                    position = 1
                # Short when TRIX turns down below zero with volume spike and price below EMA
                elif trix_val < 0 and trix_val < trix_aligned[i-1] and vol_spike and close[i] < ema_12h_val:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero or volatility spike exhaustion
            if trix_val < 0 or (trix_val < trix_aligned[i-1] and vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TRIX crosses above zero or volatility spike exhaustion
            if trix_val > 0 or (trix_val > trix_aligned[i-1] and vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0