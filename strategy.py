#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h ADX trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from oversold, price above 12h ADX trend filter (ADX>25 and +DI>-DI), and volume > 1.5x 6h average volume.
# Short when Williams %R crosses below -20 from overbought, price below 12h ADX trend filter (ADX>25 and +DI<-DI), and volume > 1.5x 6h average volume.
# Exit when Williams %R crosses above -20 for longs or below -80 for shorts (mean reversion target).
# Uses Williams %R for mean reversion signals, ADX for trend filter to avoid whipsaw, volume for confirmation.
# Target: 15-30 trades/year per symbol to stay within frequency limits.
name = "6h_WilliamsR_12hADX_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period)
    adx_period = 14
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    plus_dm = pd.Series(high_12h).diff()
    minus_dm = pd.Series(low_12h).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).sum() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Align ADX and DI to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_12h, plus_di.values)
    minus_di_aligned = align_htf_to_ltf(prices, df_12h, minus_di.values)
    
    # Get 6h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, adx_period*2)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr = williams_r[i]
        adx_val = adx_aligned[i]
        plus_di_val = plus_di_aligned[i]
        minus_di_val = minus_di_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # ADX trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long entry: Williams %R crosses above -80 from oversold, uptrend (+DI > -DI), volume confirmation
            if i > 0 and williams_r[i-1] <= -80 and wr > -80 and \
               plus_di_val > minus_di_val and strong_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 from overbought, downtrend (-DI > +DI), volume confirmation
            elif i > 0 and williams_r[i-1] >= -20 and wr < -20 and \
                 minus_di_val > plus_di_val and strong_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) or ADX weakens
            if wr >= -20 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) or ADX weakens
            if wr <= -80 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals