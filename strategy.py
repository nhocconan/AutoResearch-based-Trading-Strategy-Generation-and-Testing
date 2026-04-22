# Licensing Notice: This code is provided for educational purposes only. The author does not guarantee its suitability for live trading.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Uses smoothed SMAs (Jaw, Teeth, Lips) to identify trend direction and strength
# Long when Lips > Teeth > Jaw (bullish alignment), short when reverse
# Only trade in alignment with 1d EMA50 trend to avoid counter-trend trades
# Volume confirmation reduces false signals
# Target: 15-30 trades/year per symbol, works in bull/bear via trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams Alligator components (12h timeframe)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA = smoothed moving average (similar to RMA/Wilder's smoothing)
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA/Wilder's smoothing)"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            else:
                result[i] = np.nan
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift components as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that would look ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i]
            
            # Long: Bullish alignment + volume spike + uptrend (price > EMA50)
            if bullish and vol_spike[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + volume spike + downtrend (price < EMA50)
            elif bearish and vol_spike[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: When Alligator lines re-cross (trend weakness)
            if position == 1:
                # Exit long when Lips < Teeth (bullish alignment broken)
                if lips_shifted[i] < teeth_shifted[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short when Lips > Teeth (bearish alignment broken)
                if lips_shifted[i] > teeth_shifted[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Trend_Volume_Session"
timeframe = "12h"
leverage = 1.0