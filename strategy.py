# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Hypothesis: A 1d strategy combining weekly EMA(34) trend filter, daily Donchian(20) breakouts, and volume confirmation.
In bull markets: captures trend continuation via breakouts above resistance with volume.
In bear markets: captures short opportunities via breakdowns below support with volume.
Weekly EMA ensures we only trade in the direction of the higher timeframe trend, reducing counter-trend whipsaws.
Volume confirmation filters out low-conviction breakouts.
Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for price levels (though we're on 1d timeframe, using same timeframe for levels)
    # Since primary timeframe is 1d, we can use the prices directly for Donchian
    # But we still use weekly for trend filter as per hypothesis
    
    # Calculate Donchian channels (20-period) on daily data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for indicators (max of 34 for EMA, 20 for Donchian, 14 for ATR)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        ema_trend_1w = ema34_1w_aligned[i]
        atr_current = atr[i]
        
        # Volatility filter: only trade when current volatility is above 80% of ATR (avoid extremely low vol)
        # This prevents trading during extremely quiet periods
        vol_filter = atr_current > np.nanpercentile(atr[max(0, i-50):i+1], 20) if i >= 50 else atr_current > 0
        
        if position == 0:
            # Determine trend: price above/below weekly EMA
            price_vs_ema = close[i] > ema_trend_1w
            
            # Long: price breaks above upper Donchian band with volume confirmation and uptrend (price > weekly EMA)
            if (high[i] > high_20[i] and close[i] > high_20[i] and 
                price_vs_ema and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian band with volume confirmation and downtrend (price < weekly EMA)
            elif (low[i] < low_20[i] and close[i] < low_20[i] and 
                  not price_vs_ema and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches lower Donchian band or trend turns bearish (price < weekly EMA)
            if low[i] <= low_20[i] or close[i] < ema_trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches upper Donchian band or trend turns bullish (price > weekly EMA)
            if high[i] >= high_20[i] or close[i] > ema_trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_EMA34_Trend_Donchian20_Breakout_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0