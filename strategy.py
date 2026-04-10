#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1w volume > 1.5x 20-period average volume AND chop < 61.8 (trending regime)
# - Short when price breaks below Camarilla L3 level AND 1w volume > 1.5x 20-period average volume AND chop < 61.8 (trending regime)
# - Exit when price crosses back inside the Camarilla H3-L3 range
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels that often act as breakout points
# - Volume confirmation ensures breakouts have conviction
# - Chop filter avoids whipsaws in ranging markets

name = "1d_1w_camarilla_pivot_vol_chop_v1"
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
    
    # Pre-compute 1d OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low), etc.
    # We use H3/L3 for breakouts
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    h3 = prev_close + 1.125 * camarilla_range
    l3 = prev_close - 1.125 * camarilla_range
    h4 = prev_close + 1.5 * camarilla_range
    l4 = prev_close - 1.5 * camarilla_range
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Chopiness Index (14-period) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = true_range(high, low, prev_close)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR14) / (max_high14 - min_low14)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high14 - min_low14
    
    # Avoid division by zero
    chop = np.full_like(close, 50.0, dtype=float)  # Default to neutral
    mask = (range_14 > 0) & (~np.isnan(range_14)) & (~np.isnan(sum_tr14))
    chop[mask] = 100 * np.log10(sum_tr14[mask] / range_14[mask]) / np.log10(14)
    
    # Trending regime: chop < 61.8
    trending_regime = chop < 61.8
    
    # Align HTF indicators to 1d timeframe
    # For 1w data, we need to align volume and chop (though chop is 1d here, keeping structure)
    vol_ma_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = df_1w['volume'].values > (1.5 * vol_ma_1w)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(trending_regime[i]) or
            np.isnan(volume_spike_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND trending regime
            if (close[i] > h3[i] and 
                volume_spike_1w_aligned[i] and 
                trending_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND trending regime
            elif (close[i] < l3[i] and 
                  volume_spike_1w_aligned[i] and 
                  trending_regime[i]):
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