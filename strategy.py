#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Long when price > Alligator Jaw (13-period SMMA) and Jaw > Teeth > Lips (bullish alignment) with volume > 1.5x 20-bar average.
# Short when price < Alligator Jaw and Jaw < Teeth < Lips (bearish alignment) with volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 12h timeframe to avoid overtrading.
# Williams Alligator identifies trendless periods (all lines intertwined) and strong trends (lines diverged in order).
# Works in bull (buy aligned uptrend) and bear (sell aligned downtrend) via Alligator alignment filter.

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA34 and Alligator calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        # Calculate Williams Alligator from previous 12h bars (need 13 periods for SMMA)
        if i < 13 + start_idx:
            signals[i] = 0.0
            continue
            
        # Williams Alligator: three smoothed moving averages
        # Jaw (blue): 13-period SMMA, shifted 8 bars forward
        # Teeth (red): 8-period SMMA, shifted 5 bars forward  
        # Lips (green): 5-period SMMA, shifted 3 bars forward
        # SMMA = smoothed moving average (similar to EMA but with different smoothing)
        
        # Calculate SMMA for close prices
        def smma(source, period):
            if len(source) < period:
                return np.full_like(source, np.nan)
            result = np.full_like(source, np.nan)
            # First value is simple SMA
            result[period-1] = np.mean(source[:period])
            # Subsequent values: SMMA = (PREV_SMMA*(period-1) + PRICE) / period
            for j in range(period, len(source)):
                result[j] = (result[j-1] * (period-1) + source[j]) / period
            return result
        
        # Calculate SMMA series
        smma_close = smma(close[:i+1], 1)  # dummy call to get array
        jaw_raw = smma(close[:i+1], 13)
        teeth_raw = smma(close[:i+1], 8)
        lips_raw = smma(close[:i+1], 5)
        
        if len(jaw_raw) < 13 or len(teeth_raw) < 8 or len(lips_raw) < 5:
            signals[i] = 0.0
            continue
            
        # Get latest values (already shifted in calculation)
        jaw = jaw_raw[-1] if not np.isnan(jaw_raw[-1]) else 0
        teeth = teeth_raw[-1] if not np.isnan(teeth_raw[-1]) else 0
        lips = lips_raw[-1] if not np.isnan(lips_raw[-1]) else 0
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator alignment (Jaw > Teeth > Lips) AND price > Jaw AND EMA34 uptrend AND volume confirmation
            if (jaw > teeth and teeth > lips and 
                curr_close > jaw and 
                curr_close > curr_ema_34_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment (Jaw < Teeth < Lips) AND price < Jaw AND EMA34 downtrend AND volume confirmation
            elif (jaw < teeth and teeth < lips and 
                  curr_close < jaw and 
                  curr_close < curr_ema_34_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bearish Alligator alignment (Jaw < Teeth < Lips) OR price < Jaw (trend violation) OR EMA34 downtrend
            if (jaw < teeth and teeth < lips) or \
               (curr_close < jaw) or \
               (curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish Alligator alignment (Jaw > Teeth > Lips) OR price > Jaw (trend violation) OR EMA34 uptrend
            if (jaw > teeth and teeth > lips) or \
               (curr_close > jaw) or \
               (curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals