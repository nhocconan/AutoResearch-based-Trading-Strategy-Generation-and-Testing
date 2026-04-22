#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (13,8,13) with 1d trend filter and volume confirmation.
# Long when Jaw < Teeth < Lips (bullish alignment) + price > 1d EMA50 + volume > 1.5x avg
# Short when Jaw > Teeth > Lips (bearish alignment) + price < 1d EMA50 + volume > 1.5x avg
# Exit when alignment breaks or volume drops below average.
# Williams Alligator captures trending markets; 1d EMA filter avoids counter-trend trades.
# Volume confirmation ensures moves have participation. Target: 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams Alligator on 6h data
    # Jaw: 13-period SMMA, smoothed 8 periods ahead
    # Teeth: 8-period SMMA, smoothed 5 periods ahead  
    # Lips: 5-period SMMA, smoothed 3 periods ahead
    close = prices['close'].values
    
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply smoothing offsets
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Alligator alignment conditions
        bullish_alignment = jaw[i] < teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] > lips[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: bullish alignment + price > EMA50 + volume spike
            if bullish_alignment and price > ema50_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + price < EMA50 + volume spike
            elif bearish_alignment and price < ema50_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: alignment breaks or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish alignment breaks or volume drops
                if not bullish_alignment or vol < vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish alignment breaks or volume drops
                if not bearish_alignment or vol < vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0