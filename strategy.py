#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and ATR filter
# - Long when price breaks above Camarilla H3 level AND 1w volume > 1.5x 20-period average volume AND 1d ATR(14) < median ATR(20)
# - Short when price breaks below Camarilla L3 level AND 1w volume > 1.5x 20-period average volume AND 1d ATR(14) < median ATR(20)
# - Exit when price crosses back inside the Camarilla H3-L3 range
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels that work well on daily timeframe
# - Volume confirmation reduces false breakouts
# - ATR filter ensures we trade during low volatility periods when breakouts are more reliable

name = "1d_1w_camarilla_pivot_atr_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute daily OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute Camarilla pivot levels from previous day
    # Camarilla levels: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    # We use H3/L3 for breakout signals
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    h3 = prev_close + 1.1 * camarilla_range
    l3 = prev_close - 1.1 * camarilla_range
    
    # Pre-compute 1w volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d ATR(14) for regime filter
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # ATR regime: low volatility when current ATR < median of last 20 ATR values
    atr_median_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).median().values
    low_vol_regime = atr_1d < atr_median_20
    
    # Align HTF indicators to 1d timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1w, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(low_vol_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND low volatility regime
            if (close[i] > h3[i] and 
                volume_spike_aligned[i] and 
                low_vol_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND low volatility regime
            elif (close[i] < l3[i] and 
                  volume_spike_aligned[i] and 
                  low_vol_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside the H3-L3 range
            exit_long = (position == 1 and close[i] < h3[i])
            exit_short = (position == -1 and close[i] > l3[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals