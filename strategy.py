# 1d_Williams_Alligator_1wTrend_VolumeFilter
# Hypothesis: Williams Alligator (Jaw=13, Teeth=8, Lips=5) on daily data with weekly trend filter and volume spike confirmation.
# Alligator identifies trend strength: converging lines = ranging (avoid), diverging lines = trending (trade).
# Long when Lips > Teeth > Jaw in uptrend with volume spike; Short when Lips < Teeth < Jaw in downtrend with volume spike.
# Uses weekly trend to filter direction, reducing whipsaw in bear markets. Targets 10-25 trades/year on daily timeframe.

name = "1d_Williams_Alligator_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Williams Alligator on daily: SMAs with future shift (Jaw=13, Teeth=8, Lips=5)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get weekly EMA for trend filter (34-period)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (20-period MA on daily)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5, 34, 20) + 8  # Warmup for Alligator + shifts + weekly EMA + volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Alligator alignment: Lips < Teeth < Jaw = bearish alignment
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long entry: bullish Alligator alignment + weekly uptrend + volume spike
            if bullish_alignment and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator alignment + weekly downtrend + volume spike
            elif bearish_alignment and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator convergence (Lips <= Teeth) or weekly trend reversal
            if lips[i] <= teeth[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator convergence (Lips >= Teeth) or weekly trend reversal
            if lips[i] >= teeth[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals