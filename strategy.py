#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and ATR filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels on 1d
# - Volume filter: 1w volume > 1.5x 20-period average volume (ensures institutional participation)
# - ATR filter: 1w ATR(14) < 0.04 * price (low volatility environment for cleaner breakouts)
# - Position size: 0.30 discrete level to balance return and drawdown
# - Stoploss: 2.5x ATR(20) on 1d
# - Target: 20-60 trades/year (80-240 total over 4 years) per 1d strategy guidelines

name = "1d_1w_camarilla_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w volume spike filter
    volume_1w = df_1w['volume'].values
    avg_volume_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1w > (1.5 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike)
    
    # Pre-compute 1w ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr_1w1 = high_1w - low_1w
    tr_1w2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr_1w3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr_1w1, np.maximum(tr_1w2, tr_1w3))
    tr_1w[0] = tr_1w1[0]
    atr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_filter = (atr_14 / close_1w) < 0.04  # ATR < 4% of price
    atr_filter_aligned = align_htf_to_ltf(prices, df_1w, atr_filter)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Calculate pivot points from previous bar
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    camarilla_h3 = pivot + (range_hl * 1.1 / 4.0)
    camarilla_l3 = pivot - (range_hl * 1.1 / 4.0)
    
    # Pre-compute 1d ATR(20) for stoploss
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_20 = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_filter_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Camarilla mean reversion OR stoploss hit
            if close_1d[i] < camarilla_l3[i] or close_1d[i] < entry_price - 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: Camarilla mean reversion OR stoploss hit
            if close_1d[i] > camarilla_h3[i] or close_1d[i] > entry_price + 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Look for Camarilla breakouts with volume and volatility filters
            if vol_spike_aligned[i] and atr_filter_aligned[i]:
                # Long: price breaks above Camarilla H3
                if close_1d[i] > camarilla_h3[i]:
                    position = 1
                    entry_price = close_1d[i]
                    signals[i] = 0.30
                # Short: price breaks below Camarilla L3
                elif close_1d[i] < camarilla_l3[i]:
                    position = -1
                    entry_price = close_1d[i]
                    signals[i] = -0.30
    
    return signals