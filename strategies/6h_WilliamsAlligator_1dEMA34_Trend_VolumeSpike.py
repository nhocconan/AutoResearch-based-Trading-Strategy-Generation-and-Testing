#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Uses Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) to identify trends
# Long when Lips > Teeth > Jaw (bullish alignment) with price above Teeth, bullish 1d trend, and volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) with price below Teeth, bearish 1d trend, and volume spike
# Designed for 6h timeframe to target 15-35 trades/year per symbol.
# Works in bull/bear via multi-timeframe trend alignment and volatility-based signals.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Williams Alligator on 6h data: Jaw(13), Teeth(8), Lips(5)
    # All are SMAs with specific periods
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer, higher quality trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: bullish alignment + price above Teeth + bullish 1d trend + volume spike
            if (bullish_alignment and 
                close[i] > teeth[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + price below Teeth + bearish 1d trend + volume spike
            elif (bearish_alignment and 
                  close[i] < teeth[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator alignment breaks or trend reversal
            if position == 1:
                # Exit on bearish alignment or price below Teeth or trend reversal
                bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
                if (not bullish_alignment or 
                    close[i] < teeth[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on bullish alignment or price above Teeth or trend reversal
                bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
                if (not bearish_alignment or 
                    close[i] > teeth[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0