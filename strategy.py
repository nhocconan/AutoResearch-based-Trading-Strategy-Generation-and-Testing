#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Hypothesis: 1d strategy using weekly Camarilla pivot levels (H3/L3) with volume confirmation and ATR stoploss.
# Long: Price breaks above weekly H3, volume > 1.5x 20-period average, and ATR(14) > 0.01*close (volatility filter).
# Short: Price breaks below weekly L3, volume > 1.5x 20-period average, and ATR(14) > 0.01*close.
# Exit: Opposite pivot break (L3 for long, H3 for short) or ATR trailing stop (2.0x ATR from extreme).
# Uses weekly Camarilla for structure, volume to confirm conviction, ATR for dynamic stops.
# Target: 10-30 trades/year (40-120 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter and trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1w data for Camarilla pivots (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from 1w OHLC
    # Camarilla: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_h3 = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_l3 = close_1w - 1.1 * (high_1w - low_1w)
    
    # Align HTF Camarilla levels to 1d timeframe (wait for completed 1w bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Volatility filter: ATR > 1.0% of price (avoid low-vol chop)
        vol_filter = atr[i] > 0.01 * close[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.0*ATR from high
            if long_high > 0 and close[i] < long_high - 2.0 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price breaks below L3 (opposite pivot)
            elif low[i] < camarilla_l3_aligned[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.0*ATR from low
            if short_low > 0 and close[i] > short_low + 2.0 * atr[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            # Exit: Price breaks above H3 (opposite pivot)
            elif high[i] > camarilla_h3_aligned[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above H3, volume confirmed, and sufficient volatility
            if (high[i] > camarilla_h3_aligned[i] and volume_confirmed and vol_filter):
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            # Short entry: Price breaks below L3, volume confirmed, and sufficient volatility
            elif (low[i] < camarilla_l3_aligned[i] and volume_confirmed and vol_filter):
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals