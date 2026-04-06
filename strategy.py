#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation
# Long when Alligator jaw < teeth < lips (bullish alignment) and price > 1d EMA(50)
# Short when Alligator jaw > teeth > lips (bearish alignment) and price < 1d EMA(50)
# Uses volume > 20-period average to confirm signals
# Target: 50-150 total trades over 4 years with controlled risk in both bull and bear markets
# Uses 12h timeframe with 1d trend filter to reduce trade frequency and improve signal quality

name = "12h_williams_alligator_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams Alligator (13,8,5 SMAs with future shift)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    def smma(series, period):
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        result = np.full_like(sma, np.nan, dtype=float)
        for i in range(period-1, len(sma)):
            if i == period-1:
                result[i] = sma[i]
            else:
                result[i] = (result[i-1] * (period-1) + sma[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply Alligator shifts (jaw: +8, teeth: +5, lips: +3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set NaN for shifted positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price below 1d EMA or Alligator alignment breaks
            elif close[i] < ema_1d_aligned[i] or not (jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price above 1d EMA or Alligator alignment breaks
            elif close[i] > ema_1d_aligned[i] or not (jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if vol_filter[i]:
                # Bullish alignment: jaw < teeth < lips
                bullish = jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i]
                # Bearish alignment: jaw > teeth > lips
                bearish = jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]
                
                # Long when bullish alignment and price > 1d EMA
                if bullish and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when bearish alignment and price < 1d EMA
                elif bearish and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals