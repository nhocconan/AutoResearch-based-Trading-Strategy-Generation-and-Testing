#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Williams Alligator: Jaw (EMA13, 8-period offset), Teeth (EMA8, 5-period offset), Lips (EMA5, 3-period offset).
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > close[1] AND 1w EMA50 > close AND volume > 1.5x 20-period MA.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < close[1] AND 1w EMA50 < close AND volume > 1.5x 20-period MA.
# Exit when alignment breaks (Lips crosses Teeth or Jaw) OR 1w EMA50 crosses price.
# Uses 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year) with strict entry conditions.
# Williams Alligator identifies trend phases, 1w EMA50 filters for higher timeframe trend, volume confirms participation.
# Designed to work in both bull (strong bullish alignment) and bear (strong bearish alignment) markets.

name = "1d_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator components on 1d
    # Jaw: EMA13, 8-period offset
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    # Teeth: EMA8, 5-period offset
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    # Lips: EMA5, 3-period offset
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Williams Alligator conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # 1w EMA50 trend filter
        above_1w_ema = close[i] > ema_50_1w_aligned[i]
        below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # Volume spike condition: current 1d volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        # Price momentum: current close vs previous close
        price_up = close[i] > close[i-1]
        price_down = close[i] < close[i-1]
        
        if position == 0:
            # Long: Bullish alignment AND above 1w EMA50 AND price up AND volume spike AND session
            if bullish_alignment and above_1w_ema and price_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND below 1w EMA50 AND price down AND volume spike AND session
            elif bearish_alignment and below_1w_ema and price_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish alignment OR price crosses below 1w EMA50
            if bearish_alignment or not above_1w_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish alignment OR price crosses above 1w EMA50
            if bullish_alignment or not below_1w_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals