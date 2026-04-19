#%pip install numpy pandas
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d_1w_Keltner_Breakout_VolumeFilter
# Hypothesis: Weekly trend direction via EMA200, daily Keltner breakout with volume confirmation
# Works in bull/bear by filtering direction with weekly EMA200
# Target: 30-100 trades over 4 years (7-25/year)
name = "1d_1w_Keltner_Breakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for trend direction
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for Keltner channels
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Daily EMA20 for Keltner middle
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels: upper = EMA20 + 2*ATR, lower = EMA20 - 2*ATR
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr
    
    # Volume filter: volume > 1.5 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for weekly EMA200 and daily EMA20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        # Trend filter: only long above weekly EMA200, only short below
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Keltner band + volume + weekly uptrend
            if close[i] > keltner_upper[i] and volume_filter[i] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner band + volume + weekly downtrend
            elif close[i] < keltner_lower[i] and volume_filter[i] and weekly_downtrend:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Exit long: price closes below EMA20 (middle) or volume drops
            if close[i] < ema20[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Exit short: price closes above EMA20 (middle) or volume drops
            if close[i] > ema20[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals