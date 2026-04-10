#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and 1d volume confirmation
# - Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1w close > 1w EMA200 AND 1d volume > 1.5x 20-period average
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1w close < 1w EMA200 AND 1d volume > 1.5x 20-period average
# - Exit when Alligator lines re-cross (Lips cross Teeth) or volume drops below average
# - Uses discrete position sizing 0.25 to limit fee churn
# - Alligator identifies emerging trends; volume confirms participation; 1w EMA200 filters counter-trend trades
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1w_1d_alligator_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator SMAs (using SMA for consistency with original)
    def sma(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = sma(median_price, 13)
    teeth = sma(median_price, 8)
    lips = sma(median_price, 5)
    
    # Apply Alligator offsets (shift forward by future bars)
    jaw = np.roll(jaw, 8)   # Jaw: 13-period, shift 8 bars forward
    teeth = np.roll(teeth, 5)  # Teeth: 8-period, shift 5 bars forward
    lips = np.roll(lips, 3)    # Lips: 5-period, shift 3 bars forward
    
    # Set NaN for invalid periods due to rolling
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Pre-compute 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 200:
        multiplier = 2.0 / (200 + 1)
        ema_200_1w[199] = np.mean(close_1w[:200])  # Seed with SMA
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (close_1w[i] * multiplier) + (ema_200_1w[i-1] * (1 - multiplier))
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align HTF indicators to 12h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup for Alligator and EMA
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume[i] > (1.5 * vol_ma_20_1d_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long conditions: bullish alignment AND 1w above EMA200 AND volume confirmed
            if bullish_alignment and (close[i] > ema_200_1w_aligned[i]) and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish alignment AND 1w below EMA200 AND volume confirmed
            elif bearish_alignment and (close[i] < ema_200_1w_aligned[i]) and volume_confirmed:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator lines re-cross (Lips cross Teeth)
            lips_teeth_cross = (position == 1 and lips[i] < teeth[i]) or (position == -1 and lips[i] > teeth[i])
            # Optional: volume drops below average (weakening momentum)
            volume_weak = volume[i] < vol_ma_20_1d_aligned[i]
            
            if lips_teeth_cross or volume_weak:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and 1d volume confirmation
# - Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1w close > 1w EMA200 AND 1d volume > 1.5x 20-period average
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1w close < 1w EMA200 AND 1d volume > 1.5x 20-period average
# - Exit when Alligator lines re-cross (Lips cross Teeth) or volume drops below average
# - Uses discrete position sizing 0.25 to limit fee churn
# - Alligator identifies emerging trends; volume confirms participation; 1w EMA200 filters counter-trend trades
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1w_1d_alligator_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator SMAs (using SMA for consistency with original)
    def sma(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = sma(median_price, 13)
    teeth = sma(median_price, 8)
    lips = sma(median_price, 5)
    
    # Apply Alligator offsets (shift forward by future bars)
    jaw = np.roll(jaw, 8)   # Jaw: 13-period, shift 8 bars forward
    teeth = np.roll(teeth, 5)  # Teeth: 8-period, shift 5 bars forward
    lips = np.roll(lips, 3)    # Lips: 5-period, shift 3 bars forward
    
    # Set NaN for invalid periods due to rolling
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Pre-compute 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 200:
        multiplier = 2.0 / (200 + 1)
        ema_200_1w[199] = np.mean(close_1w[:200])  # Seed with SMA
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (close_1w[i] * multiplier) + (ema_200_1w[i-1] * (1 - multiplier))
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align HTF indicators to 12h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup for Alligator and EMA
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume[i] > (1.5 * vol_ma_20_1d_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long conditions: bullish alignment AND 1w above EMA200 AND volume confirmed
            if bullish_alignment and (close[i] > ema_200_1w_aligned[i]) and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish alignment AND 1w below EMA200 AND volume confirmed
            elif bearish_alignment and (close[i] < ema_200_1w_aligned[i]) and volume_confirmed:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator lines re-cross (Lips cross Teeth)
            lips_teeth_cross = (position == 1 and lips[i] < teeth[i]) or (position == -1 and lips[i] > teeth[i])
            # Optional: volume drops below average (weakening momentum)
            volume_weak = volume[i] < vol_ma_20_1d_aligned[i]
            
            if lips_teeth_cross or volume_weak:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals