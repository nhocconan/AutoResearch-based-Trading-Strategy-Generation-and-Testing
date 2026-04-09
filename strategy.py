#!/usr/bin/env python3
# 6h_1w_cci_reversal_v1
# Hypothesis: 6h strategy using weekly CCI extreme reversals with volume confirmation and ATR trailing stop.
# Long: Weekly CCI < -100 (oversold) with volume > 1.3x 20-period average, ATR trailing stop (2.5x ATR from low).
# Short: Weekly CCI > +100 (overbought) with volume > 1.3x 20-period average, ATR trailing stop (2.5x ATR from high).
# Exit: Opposite CCI extreme or ATR trailing stop.
# Uses weekly CCI for major reversal points, 6h for execution, volume for confirmation, ATR for dynamic stops.
# Target: 12-30 trades/year (50-120 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_cci_reversal_v1"
timeframe = "6h"
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
    
    # Get 1w data for CCI (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly CCI(20)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    tp_s = pd.Series(typical_price.values)
    tp_ma = tp_s.rolling(window=20, min_periods=20).mean().values
    
    # Mean deviation
    md = tp_s.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True).values
    # Avoid division by zero
    md = np.where(md == 0, 1e-10, md)
    
    cci = (tp_s.values - tp_ma) / (0.015 * md)
    
    # Align HTF CCI to 6h timeframe (wait for completed 1w bar)
    cci_aligned = align_htf_to_ltf(prices, df_1w, cci)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_low = 0.0   # lowest low since long entry
    short_high = 0.0 # highest high since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(cci_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Update lowest low since entry
            long_low = min(long_low, low[i]) if long_low > 0 else low[i]
            # ATR trailing stop: exit if price drops 2.5*ATR from low
            if long_low > 0 and close[i] < long_low - 2.5 * atr[i]:
                position = 0
                long_low = 0.0
                signals[i] = 0.0
            # Exit: Weekly CCI > -100 (exit oversold)
            elif cci_aligned[i] > -100:
                position = 0
                long_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update highest high since entry
            short_high = max(short_high, high[i]) if short_high > 0 else high[i]
            # ATR trailing stop: exit if price rises 2.5*ATR from high
            if short_high > 0 and close[i] > short_high + 2.5 * atr[i]:
                position = 0
                short_high = 0.0
                signals[i] = 0.0
            # Exit: Weekly CCI < +100 (exit overbought)
            elif cci_aligned[i] < 100:
                position = 0
                short_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for CCI extreme with volume confirmation
            oversold = (cci_aligned[i] < -100) and volume_confirmed
            overbought = (cci_aligned[i] > 100) and volume_confirmed
            
            if oversold:
                position = 1
                long_low = low[i]
                signals[i] = 0.25
            elif overbought:
                position = -1
                short_high = high[i]
                signals[i] = -0.25
    
    return signals