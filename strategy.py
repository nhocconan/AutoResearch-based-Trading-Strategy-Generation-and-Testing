#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + volume confirmation
# - Long when Alligator bullish (jaw < teeth < lips) AND Elder Bull Power > 0 AND 12h volume > 1.5x 20-bar avg
# - Short when Alligator bearish (jaw > teeth > lips) AND Elder Bear Power < 0 AND 12h volume > 1.5x 20-bar avg
# - Exit when Alligator reverses (jaws cross teeth) OR Elder power crosses zero
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Alligator identifies trend direction via smoothed medians
# - Elder Ray measures bull/bear power relative to EMA13
# - Volume confirmation ensures institutional participation
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: Alligator catches trends, Elder Ray filters false breakouts

name = "12h_1d_alligator_elder_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h Williams Alligator (SMMA of median price)
    # Median price = (high + low) / 2
    median_price = (prices['high'] + prices['low']) / 2
    # Smoothed Moving Average (SMMA) = EMA with alpha = 1/period
    jaw = median_price.ewm(alpha=1/13, adjust=False, min_periods=13).mean().values  # 13-period
    teeth = median_price.ewm(alpha=1/8, adjust=False, min_periods=8).mean().values    # 8-period
    lips = median_price.ewm(alpha=1/5, adjust=False, min_periods=5).mean().values     # 5-period
    
    # Alligator conditions: jaw < teeth < lips (bullish), jaw > teeth > lips (bearish)
    alligator_bullish = (jaw < teeth) & (teeth < lips)
    alligator_bearish = (jaw > teeth) & (teeth > lips)
    
    # Pre-compute 12h Elder Ray Power (relative to EMA13)
    close = prices['close'].values
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    # Actually, Elder Ray uses: Bull Power = High - EMA, Bear Power = Low - EMA
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema13
    bear_power = low - ema13
    elder_bull = bull_power > 0
    elder_bear = bear_power < 0
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    # Pre-compute 1d trend filter: close > EMA50 for bullish, close < EMA50 for bearish
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    close_gt_ema50 = close_1d > ema50_1d
    close_lt_ema50 = close_1d < ema50_1d
    
    # Align HTF indicators to 12h timeframe
    close_gt_ema50_aligned = align_htf_to_ltf(prices, df_1d, close_gt_ema50)
    close_lt_ema50_aligned = align_htf_to_ltf(prices, df_1d, close_lt_ema50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup for longest indicator
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_spike[i]) or np.isnan(close_gt_ema50_aligned[i]) or
            np.isnan(close_lt_ema50_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Alligator bullish AND Elder Bull Power > 0 AND volume spike AND 1d bullish trend
            if (alligator_bullish[i] and 
                elder_bull[i] and 
                vol_spike[i] and 
                close_gt_ema50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when Alligator bearish AND Elder Bear Power < 0 AND volume spike AND 1d bearish trend
            elif (alligator_bearish[i] and 
                  elder_bear[i] and 
                  vol_spike[i] and 
                  close_lt_ema50_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator reverses OR Elder power crosses zero
            exit_long = not alligator_bullish[i] or elder_bull[i] == False
            exit_short = not alligator_bearish[i] or elder_bear[i] == False
            
            if (position == 1 and exit_long) or (position == -1 and exit_short):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals