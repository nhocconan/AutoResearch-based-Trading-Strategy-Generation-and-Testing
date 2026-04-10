#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout + 1w ATR regime filter + volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1w ATR(14) < median ATR(20) (low volatility regime) AND volume > 1.8x 20-period average
# - Short when price breaks below Camarilla L3 level AND 1w ATR(14) < median ATR(20) AND volume > 1.8x 20-period average
# - Exit when price returns to Camarilla PIVOT level (mean reversion to equilibrium)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
# - Camarilla levels provide institutional support/resistance that work in both trending and ranging markets
# - ATR filter ensures we trade during low volatility periods when breakouts are more reliable
# - Volume confirmation reduces false breakouts

name = "1d_1w_camarilla_atr_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
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
    
    # ATR regime: low volatility when current ATR < median of last 20 ATR values
    atr_ma_20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_median_20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).median().values
    low_vol_regime = atr_1w < atr_median_20
    
    # Pre-compute 1d Camarilla levels from previous period's OHLC
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
    
    # Align HTF indicators to 1d timeframe
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1w, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pivot[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(low_vol_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND low volatility regime AND volume spike
            if (close[i] > camarilla_h3[i] and 
                low_vol_regime_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND low volatility regime AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  low_vol_regime_aligned[i] and 
                  volume_spike[i]):
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