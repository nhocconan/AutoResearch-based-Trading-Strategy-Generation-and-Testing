#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter (price > SMA50) and volume confirmation.
# Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND price > 1d SMA50 AND volume > 1.5x 24-bar average.
# Short when jaws < teeth < lips AND price < 1d SMA50 AND volume > 1.5x 24-bar average.
# Uses discrete sizing 0.25 to limit drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
# 1d SMA50 provides robust trend alignment that works in both bull (price above SMA) and bear (price below SMA).
# Williams Alligator identifies strong trending markets with clear separation of lines.
# Volume confirmation (1.5x average) ensures only high-conviction breakouts are traded.

name = "12h_WilliamsAlligator_1dSMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for SMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d SMA50 calculation
    close_1d = df_1d['close'].values
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    # 1d trend: price above/below SMA50
    price_above_sma = close > sma_50_aligned
    price_below_sma = close < sma_50_aligned
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaws: 13-period SMMA of median price, shifted 8 bars forward
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    median_price = (high + low) / 2
    
    def smma(data, period):
        """Smoothed Moving Average (SMMA) - also known as Wilder's MA"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaws_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply forward shifts as per Alligator definition
    jaws = np.full_like(jaws_raw, np.nan, dtype=float)
    teeth = np.full_like(teeth_raw, np.nan, dtype=float)
    lips = np.full_like(lips_raw, np.nan, dtype=float)
    
    # Shift jaws forward by 8 bars
    if len(jaws_raw) > 8:
        jaws[8:] = jaws_raw[:-8]
    # Shift teeth forward by 5 bars
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    # Shift lips forward by 3 bars
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Alligator conditions: jaws > teeth > lips (bullish) or jaws < teeth < lips (bearish)
    bullish_alligator = (jaws > teeth) & (teeth > lips)
    bearish_alligator = (jaws < teeth) & (teeth < lips)
    
    # Volume confirmation: current 12h volume > 1.5x 24-bar average (equivalent to 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for SMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Alligator signals
        bullish = bullish_alligator[i]
        bearish = bearish_alligator[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator AND price > 1d SMA50 AND volume confirmation
            if (bullish and 
                price_above_sma[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator AND price < 1d SMA50 AND volume confirmation
            elif (bearish and 
                  price_below_sma[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR price < 1d SMA50 (trend change)
            if (not bullish_alligator[i] or 
                not price_above_sma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR price > 1d SMA50 (trend change)
            if (not bearish_alligator[i] or 
                not price_below_sma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals