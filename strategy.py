#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ATR filter and volume confirmation
# - Primary: 1d Donchian breakout (20-period) for clear entry/exit levels
# - HTF Filter: 1w ATR(14) > 1.5x 50-period ATR MA to avoid low-volatility chop
# - Volume: 1d volume > 1.5x 20-period volume MA for institutional participation
# - Exit: Opposite Donchian breakout or ATR-based stop (via signal=0)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# - Works in bull/bear: Donchian captures trends, ATR filter avoids whipsaws in ranging markets, volume confirms strength

name = "1d_1w_donchian_atr_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w ATR(14) for volatility filter
    high_diff = high_1w - np.roll(high_1w, 1)
    low_diff = np.roll(low_1w, 1) - low_1w
    high_diff[0] = 0
    low_diff[0] = 0
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_50)
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Calculate 1d volume MA(20) for volume filter
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 1w ATR > 1.5x 50-period ATR MA
        volatile_enough = atr_14_aligned[i] > 1.5 * atr_ma_50_aligned[i]
        
        # Volume filter: current 1d volume > 1.5x 20-period volume MA
        volume_confirmed = volume[i] > 1.5 * volume_ma_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + volatility + volume
            if (close[i] > donchian_high[i] and volatile_enough and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + volatility + volume
            elif (close[i] < donchian_low[i] and volatile_enough and volume_confirmed):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price breaks opposite Donchian channel
            if position == 1:  # Long position
                if close[i] < donchian_low[i]:  # Exit when price breaks below Donchian low
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > donchian_high[i]:  # Exit when price breaks above Donchian high
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals