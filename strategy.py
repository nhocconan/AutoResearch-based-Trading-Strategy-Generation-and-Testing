#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d volume confirmation and 1w trend filter
# - Long when price touches Camarilla L3 level with 1d volume spike and 1w uptrend (close > EMA50)
# - Short when price touches Camarilla H3 level with 1d volume spike and 1w downtrend (close < EMA50)
# - Uses 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# - Camarilla levels calculated from prior 1d OHLC: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
# - 1d volume > 1.8x 20-period average confirms reversal strength
# - 1w EMA50 filter ensures trading with weekly trend direction
# - Discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14) from entry

name = "12h_1d_1w_camarilla_volume_trend_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter (backup if 1w insufficient)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume confirmation: > 1.8x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h Camarilla levels from prior 1d OHLC
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4, L4 = close - 1.5*(high-low)
    camarilla_high = close_1d + 1.1 * (high_1d - low_1d) / 4  # H3 level
    camarilla_low = close_1d - 1.1 * (high_1d - low_1d) / 4   # L3 level
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price breaks above Camarilla H3 (failed reversal)
            if (prices['close'].iloc[i] > entry_price + 2.5 * entry_atr or 
                prices['close'].iloc[i] > camarilla_high_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price breaks below Camarilla L3 (failed reversal)
            if (prices['close'].iloc[i] < entry_price - 2.5 * entry_atr or 
                prices['close'].iloc[i] < camarilla_low_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touch with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: price touches Camarilla L3 in weekly uptrend
                if (prices['low'].iloc[i] <= camarilla_low_aligned[i] and 
                    prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = 0.25
                # Short signal: price touches Camarilla H3 in weekly downtrend
                elif (prices['high'].iloc[i] >= camarilla_high_aligned[i] and 
                      prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = -0.25
    
    return signals