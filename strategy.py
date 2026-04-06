#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1w trend filter and volume confirmation
# Williams Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3)
# Long when: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA(50) AND volume > 2x average
# Short when: Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA(50) AND volume > 2x average
# Exit when: Alligator lines cross in opposite direction OR price crosses 1w EMA(50)
# Target: 50-150 trades over 4 years by requiring strong alignment and volume confirmation

name = "6h_williams_alligator_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)   # Jaw shifted 8 bars forward
    teeth = np.roll(teeth, 5) # Teeth shifted 5 bars forward
    lips = np.roll(lips, 3)   # Lips shifted 3 bars forward
    
    # First 8, 5, 3 values become NaN due to roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Alligator bearish crossover OR price < 1w EMA(50)
            if (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Alligator bullish crossover OR price > 1w EMA(50)
            if (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment + trend filter + volume
            # Bullish: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            if volume[i] > volume_threshold[i]:
                if bullish_alignment and close[i] > ema_50_aligned[i]:
                    # Bullish alignment above weekly EMA
                    signals[i] = 0.25
                    position = 1
                elif bearish_alignment and close[i] < ema_50_aligned[i]:
                    # Bearish alignment below weekly EMA
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1w trend filter and volume confirmation
# Williams Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3)
# Long when: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA(50) AND volume > 2x average
# Short when: Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA(50) AND volume > 2x average
# Exit when: Alligator lines cross in opposite direction OR price crosses 1w EMA(50)
# Target: 50-150 trades over 4 years by requiring strong alignment and volume confirmation

name = "6h_williams_alligator_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)   # Jaw shifted 8 bars forward
    teeth = np.roll(teeth, 5) # Teeth shifted 5 bars forward
    lips = np.roll(lips, 3)   # Lips shifted 3 bars forward
    
    # First 8, 5, 3 values become NaN due to roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Alligator bearish crossover OR price < 1w EMA(50)
            if (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Alligator bullish crossover OR price > 1w EMA(50)
            if (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment + trend filter + volume
            # Bullish: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            if volume[i] > volume_threshold[i]:
                if bullish_alignment and close[i] > ema_50_aligned[i]:
                    # Bullish alignment above weekly EMA
                    signals[i] = 0.25
                    position = 1
                elif bearish_alignment and close[i] < ema_50_aligned[i]:
                    # Bearish alignment below weekly EMA
                    signals[i] = -0.25
                    position = -1
    
    return signals