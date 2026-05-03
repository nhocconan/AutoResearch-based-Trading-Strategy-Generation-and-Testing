#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume spike
# Long when price > Alligator Jaw + volume spike + price > 1w EMA50
# Short when price < Alligator Jaw + volume spike + price < 1w EMA50
# Uses Alligator (SMAs: Jaw=13, Teeth=8, Lips=5) from prior 1d to avoid look-ahead
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Designed for low trade frequency (7-25/year on 1d) to minimize fee drag
# Works in bull (breakouts with trend) and bear (breakdowns with trend) markets

name = "1d_WilliamsAlligator_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator and 1w data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 13 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA (smoothed moving average) of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (df_1d['high'] + df_1d['low']) / 2.0
    
    # SMMA calculation (smoothed moving average)
    def smma(series, period):
        sma = series.rolling(window=period, min_periods=period).mean()
        # Convert to SMMA: first value is SMA, then recursive smoothing
        smma_vals = np.full_like(series, np.nan, dtype=float)
        if len(sma) >= period:
            smma_vals[period-1] = sma.iloc[period-1]
            for i in range(period, len(sma)):
                if not np.isnan(sma.iloc[i]) and not np.isnan(smma_vals[i-1]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + sma.iloc[i]) / period
        return smma_vals.values
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Shift the lines (Jaw: 8, Teeth: 5, Lips: 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # For Alligator, we typically use the Jaw as the main trend indicator
    # When price > Jaw and Jaw > Teeth > Lips = bullish
    # When price < Jaw and Jaw < Teeth < Lips = bearish
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    
    # Calculate EMA(50) on 1w for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (2.0x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20, 13+8)  # 1w EMA(50), volume MA(20), Alligator Jaw(13)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Jaw + volume spike + price > 1w EMA50
            if (close[i] > jaw_aligned[i] and volume_spike[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Jaw + volume spike + price < 1w EMA50
            elif (close[i] < jaw_aligned[i] and volume_spike[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Jaw or price < 1w EMA50
            if (close[i] < jaw_aligned[i] or close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Jaw or price > 1w EMA50
            if (close[i] > jaw_aligned[i] or close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals