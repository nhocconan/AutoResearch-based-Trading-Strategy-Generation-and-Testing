#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation.
# Long when Alligator jaws (13-period SMMA) crosses above teeth (8-period SMMA) AND price > 1w EMA50 AND volume > 1.5x 24-bar average.
# Short when jaws cross below teeth AND price < 1w EMA50 AND volume > 1.5x 24-bar average.
# Uses discrete sizing 0.25 to limit drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
# Williams Alligator provides smooth trend identification with built-in lag to reduce whipsaws.
# 1w EMA50 offers robust multi-week trend alignment that works in both bull (price above EMA) and bear (price below EMA).
# Volume confirmation ensures only high-conviction breakouts are traded.

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # 1w trend: price above/below EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Williams Alligator calculation (1d timeframe)
    # Jaws: 13-period SMMA of median price, shifted 8 bars forward
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    median_price = (high + low) / 2.0
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to RMA/Wilder's smoothing
    def smma(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    smma_5 = smma(median_price, 5)
    smma_8 = smma(median_price, 8)
    smma_13 = smma(median_price, 13)
    
    # Apply Alligator shifts (jaw: shift 8, teeth: shift 5, lips: shift 3)
    jaws = np.full_like(smma_13, np.nan)
    teeth = np.full_like(smma_8, np.nan)
    lips = np.full_like(smma_5, np.nan)
    
    # Shift forward: jaw value at i comes from smma_13[i+8], etc.
    for i in range(len(smma_13)):
        if i + 8 < len(smma_13):
            jaws[i + 8] = smma_13[i]
        if i + 5 < len(smma_8):
            teeth[i + 5] = smma_8[i]
        if i + 3 < len(smma_5):
            lips[i + 3] = smma_5[i]
    
    # Volume confirmation: current 1d volume > 1.5x 24-bar average (equivalent to ~1 month)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for SMMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
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
        jaw_above_teeth = jaws[i] > teeth[i]  # bullish alignment
        jaw_below_teeth = jaws[i] < teeth[i]  # bearish alignment
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: jaws cross above teeth AND price > 1w EMA50 AND volume confirmation
            if (jaw_above_teeth and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: jaws cross below teeth AND price < 1w EMA50 AND volume confirmation
            elif (jaw_below_teeth and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: jaws cross below teeth (trend change) OR price < 1w EMA50 (trend filter fail)
            if (jaw_below_teeth or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: jaws cross above teeth (trend change) OR price > 1w EMA50 (trend filter fail)
            if (jaw_above_teeth or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals