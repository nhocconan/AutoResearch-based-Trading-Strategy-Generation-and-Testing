#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter
# Long when: Alligator jaws < teeth < lips (bullish alignment), Elder Bull Power > 0, and price > 1d EMA34
# Short when: Alligator jaws > teeth > lips (bearish alignment), Elder Bear Power < 0, and price < 1d EMA34
# Uses 1d EMA34 for higher timeframe trend alignment (matches experiment HTF)
# Williams Alligator (SMAs: jaws=13, teeth=8, lips=5) identifies trend direction and avoids chop
# Elder Ray (Bull/Bear Power = EMA13 - high/low) measures trend strength relative to EMA13
# Discrete position sizing (0.25) to minimize fee churn
# Designed for low trade frequency (12-37/year on 6h) to avoid fee drag
# Works in bull (trend continuation) and bear (trend continuation) markets

name = "6h_Alligator_ElderRay_1dEMA34_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_ = prices['open'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h
    # Jaws: SMA(13) of median price, shifted 8 bars
    # Teeth: SMA(8) of median price, shifted 5 bars
    # Lips: SMA(5) of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate Elder Ray on 6h
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 13, 8) + 1  # EMA(34) + Alligator jaws(13) warmup
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaws[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish alignment, Bull Power > 0, price > 1d EMA34
            if (jaws[i] < teeth[i] and teeth[i] < lips[i] and 
                bull_power[i] > 0 and close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish alignment, Bear Power < 0, price < 1d EMA34
            elif (jaws[i] > teeth[i] and teeth[i] > lips[i] and 
                  bear_power[i] < 0 and close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment turns bearish or price < 1d EMA34
            if (jaws[i] > teeth[i] or teeth[i] > lips[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment turns bullish or price > 1d EMA34
            if (jaws[i] < teeth[i] or teeth[i] < lips[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals