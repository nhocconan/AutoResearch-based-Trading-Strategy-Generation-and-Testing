#!/usr/bin/env python3
"""
1d Williams Alligator with 1w EMA50 Trend Filter and Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trendless markets and trend initiation.
When price is outside the Alligator's mouth (all lines aligned) and aligned with 1w EMA50 trend,
confirmed by volume spike, it captures strong directional moves. Designed for 1d timeframe to target
7-25 trades/year (30-100 over 4 years) by requiring confluence of Alligator alignment, 1w EMA50 trend,
and volume confirmation, reducing overtrading and fee drag. Works in bull (long when lips>teeth>jaw) 
and bear (short when lips<teeth<jaw) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Williams Alligator on primary timeframe (1d)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    close_series = pd.Series(close)
    
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = series.rolling(window=period, min_periods=period).mean()
        # SMMA: first value is SMA, then recursive smoothing
        result = np.full(len(series), np.nan)
        if len(series) >= period:
            result[period-1] = sma.iloc[period-1]
            for i in range(period, len(series)):
                result[i] = (result[i-1] * (period-1) + series.iloc[i]) / period
        return result
    
    jaw = smma(close_series, 13)
    teeth = smma(close_series, 8)
    lips = smma(close_series, 5)
    
    # Shift forward: jaw+8, teeth+5, lips+3
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator shifts and EMA50
    start_idx = max(13, 50)  # Alligator jaw period, EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: check if lines are properly ordered (trending) or intertwined (ranging)
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        
        # Bullish alignment: Lips > Teeth > Jaw (alligator mouth opening up)
        bullish_aligned = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw (alligator mouth opening down)
        bearish_aligned = lips_val < teeth_val and teeth_val < jaw_val
        # Trend filter: price relative to 1w EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + trend + volume
            # Long: bullish Alligator alignment AND bullish bias AND volume spike
            long_entry = bullish_aligned and bullish_bias and vol_spike
            # Short: bearish Alligator alignment AND bearish bias AND volume spike
            short_entry = bearish_aligned and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator loses bullish alignment OR price crosses below teeth (reversion)
            if not bullish_aligned or curr_close < teeth_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator loses bearish alignment OR price crosses above teeth (reversion)
            if not bearish_aligned or curr_close > teeth_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0