#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + 12h Williams Alligator combination for trend filtering
# - Long when 6h ADX > 25 (trending) AND price > Alligator jaws (12h) AND 6h close > 6h open (bullish bar)
# - Short when 6h ADX > 25 (trending) AND price < Alligator jaws (12h) AND 6h close < 6h open (bearish bar)
# - Exit when ADX < 20 (trend weak) OR price crosses Alligator teeth (12h)
# - Uses discrete position sizing 0.25 to limit fee churn
# - ADX filters for strong trends, Alligator confirms direction and dynamic support/resistance
# - Works in both bull (trend continuation) and bear (trend continuation down) markets
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_12h_adx_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Pre-compute 6h ADX (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    # +DM and -DM
    up_move = np.zeros_like(high)
    down_move = np.zeros_like(high)
    up_move[0] = 0
    down_move[0] = 0
    for i in range(1, len(high)):
        up_move[i] = high[i] - high[i-1]
        down_move[i] = low[i-1] - low[i]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[1:period])
        # Wilder's smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Pre-compute 12h Williams Alligator (Jaws=TEETH=LIPS SMMA)
    # Alligator: Jaws (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # SMMA: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    jaws = smma(median_price_12h, 13)  # Jaws: 13-period
    teeth = smma(median_price_12h, 8)   # Teeth: 8-period
    lips = smma(median_price_12h, 5)    # Lips: 5-period
    
    # Align Alligator lines to 6h timeframe (no additional delay needed for SMMA)
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Alligator Jaw is the main trend indicator (we'll use it as the dynamic support/resistance)
    # Alligator is sleeping when jaws, teeth, lips are intertwined (no clear trend)
    # Alligator is awake when lines are separated in order (Uptrend: Lips > Teeth > Jaws, Downtrend: Lips < Teeth < Jaws)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup for ADX
        # Skip if any required data is invalid
        if (np.isnan(adx[i]) or np.isnan(jaws_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator awake and trending conditions
            # Uptrend: Lips > Teeth > Jaws
            # Downtrend: Lips < Teeth < Jaws
            alligator_uptrend = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaws_aligned[i])
            alligator_downtrend = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaws_aligned[i])
            
            # Strong trend filter
            strong_trend = adx[i] > 25
            
            # Bar direction
            bullish_bar = close[i] > open_price[i]
            bearish_bar = close[i] < open_price[i]
            
            # Long conditions: strong uptrend + bullish bar + price above jaws
            if strong_trend and alligator_uptrend and bullish_bar and close[i] > jaws_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: strong downtrend + bearish bar + price below jaws
            elif strong_trend and alligator_downtrend and bearish_bar and close[i] < jaws_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: trend weak OR price crosses Alligator teeth
            weak_trend = adx[i] < 20
            
            # For long: exit if price crosses below teeth (support broken)
            # For short: exit if price crosses above teeth (resistance broken)
            long_exit = (position == 1 and (weak_trend or close[i] < teeth_aligned[i]))
            short_exit = (position == -1 and (weak_trend or close[i] > teeth_aligned[i]))
            
            if long_exit or short_exit:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + 12h Williams Alligator combination for trend filtering
# - Long when 6h ADX > 25 (trending) AND price > Alligator jaws (12h) AND 6h close > 6h open (bullish bar)
# - Short when 6h ADX > 25 (trending) AND price < Alligator jaws (12h) AND 6h close < 6h open (bearish bar)
# - Exit when ADX < 20 (trend weak) OR price crosses Alligator teeth (12h)
# - Uses discrete position sizing 0.25 to limit fee churn
# - ADX filters for strong trends, Alligator confirms direction and dynamic support/resistance
# - Works in both bull (trend continuation) and bear (trend continuation down) markets
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_12h_adx_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Pre-compute 6h ADX (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    # +DM and -DM
    up_move = np.zeros_like(high)
    down_move = np.zeros_like(high)
    up_move[0] = 0
    down_move[0] = 0
    for i in range(1, len(high)):
        up_move[i] = high[i] - high[i-1]
        down_move[i] = low[i-1] - low[i]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[1:period])
        # Wilder's smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Pre-compute 12h Williams Alligator (Jaws=TEETH=LIPS SMMA)
    # Alligator: Jaws (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # SMMA: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    jaws = smma(median_price_12h, 13)  # Jaws: 13-period
    teeth = smma(median_price_12h, 8)   # Teeth: 8-period
    lips = smma(median_price_12h, 5)    # Lips: 5-period
    
    # Align Alligator lines to 6h timeframe (no additional delay needed for SMMA)
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Alligator Jaw is the main trend indicator (we'll use it as the dynamic support/resistance)
    # Alligator is sleeping when jaws, teeth, lips are intertwined (no clear trend)
    # Alligator is awake when lines are separated in order (Uptrend: Lips > Teeth > Jaws, Downtrend: Lips < Teeth < Jaws)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup for ADX
        # Skip if any required data is invalid
        if (np.isnan(adx[i]) or np.isnan(jaws_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator awake and trending conditions
            # Uptrend: Lips > Teeth > Jaws
            # Downtrend: Lips < Teeth < Jaws
            alligator_uptrend = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaws_aligned[i])
            alligator_downtrend = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaws_aligned[i])
            
            # Strong trend filter
            strong_trend = adx[i] > 25
            
            # Bar direction
            bullish_bar = close[i] > open_price[i]
            bearish_bar = close[i] < open_price[i]
            
            # Long conditions: strong uptrend + bullish bar + price above jaws
            if strong_trend and alligator_uptrend and bullish_bar and close[i] > jaws_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: strong downtrend + bearish bar + price below jaws
            elif strong_trend and alligator_downtrend and bearish_bar and close[i] < jaws_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: trend weak OR price crosses Alligator teeth
            weak_trend = adx[i] < 20
            
            # For long: exit if price crosses below teeth (support broken)
            # For short: exit if price crosses above teeth (resistance broken)
            long_exit = (position == 1 and (weak_trend or close[i] < teeth_aligned[i]))
            short_exit = (position == -1 and (weak_trend or close[i] > teeth_aligned[i]))
            
            if long_exit or short_exit:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals