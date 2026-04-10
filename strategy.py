#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume confirmation + 1w ATR regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average AND 1w ATR(14) < 50-period median ATR (low volatility regime)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average AND 1w ATR(14) < 50-period median ATR
# - Exit when price returns to Camarilla PIVOT level (mean reversion to equilibrium)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla levels provide institutional support/resistance that work in both trending and ranging markets
# - Volume confirmation reduces false breakouts
# - ATR regime filter ensures we trade during low volatility periods when breakouts are more reliable

name = "12h_1d_1w_camarilla_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 14:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w ATR(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1w = np.zeros_like(tr)
    atr_1w[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # ATR regime: low volatility when current ATR < median of last 50 ATR values
    atr_median_50 = pd.Series(atr_1w).rolling(window=50, min_periods=50).median().values
    low_vol_regime = atr_1w < atr_median_50
    
    # Pre-compute 12h Camarilla levels from previous period's OHLC
    # Camarilla levels use previous period's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h3 = pivot + (range_hl * 1.1 / 4)
    camarilla_l3 = pivot - (range_hl * 1.1 / 4)
    camarilla_h4 = pivot + (range_hl * 1.1 / 2)
    camarilla_l4 = pivot - (range_hl * 1.1 / 2)
    
    # Align HTF indicators to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1w, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pivot[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(low_vol_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND volume spike AND low volatility regime
            if (close[i] > camarilla_h3[i] and 
                volume_spike_aligned[i] and 
                low_vol_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND volume spike AND low volatility regime
            elif (close[i] < camarilla_l3[i] and 
                  volume_spike_aligned[i] and 
                  low_vol_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit when price returns to pivot level (mean reversion to equilibrium)
            exit_long = (position == 1 and close[i] <= pivot[i])
            exit_short = (position == -1 and close[i] >= pivot[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals