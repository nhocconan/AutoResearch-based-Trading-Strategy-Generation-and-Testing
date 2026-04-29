#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume spike confirmation
# Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips)
# Jaw (blue) = 13-period SMMA, Teeth (red) = 8-period SMMA, Lips (green) = 5-period SMMA
# Long when Lips > Teeth > Jaw (bullish alignment) + price > 1w EMA50 + volume > 2.0x 20-period average
# Short when Lips < Teeth < Jaw (bearish alignment) + price < 1w EMA50 + volume > 2.0x 20-period average
# Alligator identifies trend absence (sleeping), formation (awakening), and trend (eating)
# Works in bull markets via eating uptrends, in bear markets via eating downtrends
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe

name = "1d_WilliamsAlligator_1wEMA50_VolumeSpike"
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
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Smoothed Moving Average (SMMA) for Williams Alligator
    # SMMA is similar to EMA but with different smoothing: SMMA_t = (SMMA_{t-1} * (period-1) + price_t) / period
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Williams Alligator components
    jaw = smma(close, 13)  # Jaw (blue) - 13-period SMMA
    teeth = smma(close, 8)  # Teeth (red) - 8-period SMMA
    lips = smma(close, 5)   # Lips (green) - 5-period SMMA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 50, 20)  # Jaw, 1w EMA50, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits and trailing logic
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if curr_lips <= curr_teeth or curr_teeth <= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if curr_lips >= curr_teeth or curr_teeth >= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bullish alignment (Lips > Teeth > Jaw) + uptrend + volume confirmation
            if (i > start_idx and 
                curr_lips > curr_teeth and 
                curr_teeth > curr_jaw and 
                curr_close > curr_ema_1w and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment (Lips < Teeth < Jaw) + downtrend + volume confirmation
            elif (i > start_idx and 
                  curr_lips < curr_teeth and 
                  curr_teeth < curr_jaw and 
                  curr_close < curr_ema_1w and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals