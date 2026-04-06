#!/usr/bin/env python3
"""
6h Bollinger Band Squeeze + 12h Trend + Volume
Hypothesis: Low volatility squeezes precede breakouts. Use 12h trend direction to filter breakout entries.
Long when BB squeeze (bandwidth < 20th percentile) breaks upward with volume in 12h uptrend.
Short when BB squeeze breaks downward with volume in 12h downtrend.
Works in both bull and bear markets by capturing volatility expansion after contraction.
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14439_6h_bb_squeeze_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA20 trend
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_mult = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_mult * bb_std
    lower = sma - bb_mult * bb_std
    bandwidth = (upper - lower) / sma
    
    # Bandwidth percentile (20th) for squeeze detection
    bandwidth_series = pd.Series(bandwidth)
    bw_percentile = bandwidth_series.rolling(window=50, min_periods=20).quantile(0.20).values
    squeeze = bandwidth < bw_percentile
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(bb_period, 50)  # BB period and percentile window
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(sma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(bw_percentile[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below middle OR trend reverses OR stoploss
            if (close[i] < sma[i] or 
                (close[i] < ema_12h_aligned[i]) or  # 12h trend turned down
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above middle OR trend reverses OR stoploss
            if (close[i] > sma[i] or 
                (close[i] > ema_12h_aligned[i]) or  # 12h trend turned up
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: squeeze breakout with volume and 12h trend alignment
            long_squeeze = squeeze[i-1] and not squeeze[i]  # squeeze just released
            breakout_up = close[i] > upper[i]
            breakout_down = close[i] < lower[i]
            
            # 12h trend filter
            trend_up = close[i] > ema_12h_aligned[i]
            trend_down = close[i] < ema_12h_aligned[i]
            
            if long_squeeze and breakout_up and vol_filter[i] and trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif long_squeeze and breakout_down and vol_filter[i] and trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Bollinger Band Squeeze + 12h Trend + Volume
Hypothesis: Low volatility squeezes precede breakouts. Use 12h trend direction to filter breakout entries.
Long when BB squeeze (bandwidth < 20th percentile) breaks upward with volume in 12h uptrend.
Short when BB squeeze breaks downward with volume in 12h downtrend.
Works in both bull and bear markets by capturing volatility expansion after contraction.
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14439_6h_bb_squeeze_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA20 trend
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_mult = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_mult * bb_std
    lower = sma - bb_mult * bb_std
    bandwidth = (upper - lower) / sma
    
    # Bandwidth percentile (20th) for squeeze detection
    bandwidth_series = pd.Series(bandwidth)
    bw_percentile = bandwidth_series.rolling(window=50, min_periods=20).quantile(0.20).values
    squeeze = bandwidth < bw_percentile
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(bb_period, 50)  # BB period and percentile window
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(sma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(bw_percentile[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below middle OR trend reverses OR stoploss
            if (close[i] < sma[i] or 
                (close[i] < ema_12h_aligned[i]) or  # 12h trend turned down
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above middle OR trend reverses OR stoploss
            if (close[i] > sma[i] or 
                (close[i] > ema_12h_aligned[i]) or  # 12h trend turned up
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: squeeze breakout with volume and 12h trend alignment
            long_squeeze = squeeze[i-1] and not squeeze[i]  # squeeze just released
            breakout_up = close[i] > upper[i]
            breakout_down = close[i] < lower[i]
            
            # 12h trend filter
            trend_up = close[i] > ema_12h_aligned[i]
            trend_down = close[i] < ema_12h_aligned[i]
            
            if long_squeeze and breakout_up and vol_filter[i] and trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif long_squeeze and breakout_down and vol_filter[i] and trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>