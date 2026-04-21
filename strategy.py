#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + 1week Trend
# Uses Williams Alligator (13,8,5 SMAs) for trend direction, Elder Ray (13-period) for bull/bear power,
# and 1-week EMA (34-period) for higher timeframe trend confirmation.
# Trades only when all three align: price above/below Alligator jaws, corresponding Elder Ray power positive,
# and price above/below weekly EMA34. Filters out weak signals and reduces whipsaw.
# Target: 20-40 trades/year by requiring triple confirmation.
# Works in both bull and bear markets as it follows the higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Williams Alligator: 13,8,5 period SMAs of median price
    median_price = (prices['high'] + prices['low']) / 2
    jaw = median_price.rolling(window=13, min_periods=13).mean()  # 13-period
    teeth = median_price.rolling(window=8, min_periods=8).mean()   # 8-period
    lips = median_price.rolling(window=5, min_periods=5).mean()    # 5-period
    
    # Elder Ray: 13-period EMA of high and low
    ema13 = median_price.ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = prices['high'] - ema13
    bear_power = ema13 - prices['low']
    
    # Get 1-week data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Weekly EMA34 on close
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if (np.isnan(jaw.iloc[i]) or np.isnan(teeth.iloc[i]) or np.isnan(lips.iloc[i]) or
            np.isnan(bull_power.iloc[i]) or np.isnan(bear_power.iloc[i]) or
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = prices['close'].iloc[i]
        jaw_val = jaw.iloc[i]
        teeth_val = teeth.iloc[i]
        lips_val = lips.iloc[i]
        bull_val = bull_power.iloc[i]
        bear_val = bear_power.iloc[i]
        weekly_ema = ema34_1w_aligned[i]
        
        # Alligator alignment: check if jaws, teeth, lips are properly ordered
        # For uptrend: lips > teeth > jaw
        # For downtrend: lips < teeth < jaw
        is_uptrend_aligned = lips_val > teeth_val and teeth_val > jaw_val
        is_downtrend_aligned = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long conditions: price above lips, bull power positive, above weekly EMA, uptrend aligned
            if (price > lips_val and bull_val > 0 and price > weekly_ema and is_uptrend_aligned):
                signals[i] = 0.25
                position = 1
            # Short conditions: price below lips, bear power positive, below weekly EMA, downtrend aligned
            elif (price < lips_val and bear_val > 0 and price < weekly_ema and is_downtrend_aligned):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below lips OR bull power turns negative
                if price < lips_val or bull_val <= 0:
                    exit_signal = True
            elif position == -1:  # short position
                # Exit if price crosses above lips OR bear power turns negative
                if price > lips_val or bear_val <= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsAlligator_ElderRay_1weekTrend"
timeframe = "1d"
leverage = 1.0