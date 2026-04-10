#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator: Jaw (13-period SMMA, 8-bar offset), Teeth (8-period SMMA, 5-bar offset), Lips (5-period SMMA, 3-bar offset)
# - Long when Lips > Teeth > Jaw (bullish alignment) in 1d uptrend (close > EMA50) with volume spike
# - Short when Lips < Teeth < Jaw (bearish alignment) in 1d downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) or Alligator alignment breaks
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Alligator catches trends early; volume confirms strength; 1d EMA50 filters counter-trend noise

name = "12h_1d_alligator_volume_trend_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d Williams Alligator: SMMA (Smoothed Moving Average)
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = np.zeros_like(data)
        sma[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    # Jaw: 13-period SMMA, 8-bar offset
    jaw_raw = smma(close_1d, 13)
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA, 5-bar offset
    teeth_raw = smma(close_1d, 8)
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, 3-bar offset
    lips_raw = smma(close_1d, 5)
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or Alligator alignment breaks (Lips <= Teeth)
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                lips_aligned[i] <= teeth_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or Alligator alignment breaks (Lips >= Teeth)
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                lips_aligned[i] >= teeth_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Bullish alignment: Lips > Teeth > Jaw
                if (lips_aligned[i] > teeth_aligned[i] and 
                    teeth_aligned[i] > jaw_aligned[i] and
                    prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = 0.25
                # Bearish alignment: Lips < Teeth < Jaw
                elif (lips_aligned[i] < teeth_aligned[i] and 
                      teeth_aligned[i] < jaw_aligned[i] and
                      prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = -0.25
    
    return signals