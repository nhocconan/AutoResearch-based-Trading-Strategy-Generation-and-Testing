#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price closes above Camarilla H3 level AND 1d volume > 2.0x 20-period average volume AND 1d chop > 61.8 (range regime)
# - Short when price closes below Camarilla L3 level AND 1d volume > 2.0x 20-period average volume AND 1d chop > 61.8 (range regime)
# - Exit when price crosses Camarilla H4/L4 levels (strong reversal) or chop < 38.2 (trend regime)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots work well in ranging markets (chop > 61.8) which suits 2025 bearish conditions
# - Volume confirmation reduces false breakouts
# - Chop filter ensures we only trade in ranging regimes where mean reversion works

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Pre-compute 12h Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    #           L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    # We use previous day's range to calculate today's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_ = prev_high - prev_low
    camarilla_h3 = prev_close + 1.1 * range_
    camarilla_l3 = prev_close - 1.1 * range_
    camarilla_h4 = prev_close + 1.5 * range_
    camarilla_l4 = prev_close - 1.5 * range_
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Pre-compute 1d Chop Index (Ehler's Chopiness Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Chop Index = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    def rolling_sum(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.sum(arr[i - window + 1:i + 1])
        return result
    
    sum_tr_14 = rolling_sum(tr, 14)
    chop = 100 * np.log10(sum_tr_14 / (atr_1d * 14)) / np.log10(14)
    
    # Chop regime: range when chop > 61.8, trend when chop < 38.2
    chop_regime_range = chop > 61.8
    chop_regime_trend = chop < 38.2
    
    # Align HTF indicators to 12h timeframe
    chop_regime_range_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_range)
    chop_regime_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_trend)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(chop_regime_range_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price closes above H3 AND range regime AND volume spike
            if (close[i] > camarilla_h3[i] and 
                chop_regime_range_aligned[i] and 
                volume_spike_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price closes below L3 AND range regime AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  chop_regime_range_aligned[i] and 
                  volume_spike_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: 
            # 1. Price crosses H4/L4 (strong reversal)
            # 2. Chop regime shifts to trend (chop < 38.2)
            exit_long = (position == 1 and (close[i] > camarilla_h4[i] or chop_regime_trend_aligned[i]))
            exit_short = (position == -1 and (close[i] < camarilla_l4[i] or chop_regime_trend_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals