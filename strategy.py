#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter
# Uses 6h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Williams Alligator (jaw/teeth/lips) identifies trend absence/presence via convergence/divergence
# Elder Ray (Bull Power/Bear Power) measures buying/selling pressure relative to EMA13
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend entries
# Works in bull (Alligator eating up, Bull Power > 0) and bear (Alligator eating down, Bear Power < 0) markets
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure

name = "6h_WilliamsAlligator_ElderRay_1dEMA34_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h data for Williams Alligator (SMAs of median price)
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Williams Alligator: Jaw (13-period, 8-bar shift), Teeth (8-period, 5-bar shift), Lips (5-period, 3-bar shift)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for Alligator (max shift 8) and EMA13
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend bias
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        # Williams Alligator conditions
        # Alligator sleeping (convergence): jaws, teeth, lips intertwined -> no trend
        alligator_sleeping = (abs(jaw[i] - teeth[i]) < (abs(jaw[i]) * 0.001)) and \
                            (abs(teeth[i] - lips[i]) < (abs(teeth[i]) * 0.001))
        
        # Alligator awakening (divergence): lines separating -> trend emerging
        # Bullish: Lips > Teeth > Jaw (green alignment)
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish: Lips < Teeth < Jaw (red alignment)
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias and alligator_bullish and bull_power[i] > 0:
                # Long: 1d uptrend + Alligator eating up + Bull Power positive
                signals[i] = 0.25
                position = 1
            elif bearish_bias and alligator_bearish and bear_power[i] < 0:
                # Short: 1d downtrend + Alligator eating down + Bear Power negative
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator sleeping (loss of trend) or Bear Power turns negative
            if alligator_sleeping or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator sleeping (loss of trend) or Bull Power turns positive
            if alligator_sleeping or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals