#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter
# - Long when Williams %R(14) < -80 (oversold) AND 1d EMA50 > EMA200 (bullish trend)
# - Short when Williams %R(14) > -20 (overbought) AND 1d EMA50 < EMA200 (bearish trend)
# - Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams %R captures short-term reversals; 1d EMA filter ensures we trade with the higher timeframe trend
# - Mean reversion works well in ranging markets which dominate BTC/ETH price action
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_williamsr_meanreversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute 6h Williams %R (14-period)
    def highest_high(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def lowest_low(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    hh_6h = highest_high(high, 14)
    ll_6h = lowest_low(low, 14)
    williams_r = np.full_like(close, np.nan, dtype=float)
    for i in range(13, len(close)):
        if hh_6h[i] > ll_6h[i]:
            williams_r[i] = (hh_6h[i] - close[i]) / (hh_6h[i] - ll_6h[i]) * -100
        else:
            williams_r[i] = -50.0
    
    # Pre-compute 1d EMA50 and EMA200
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND 1d bullish trend (EMA50 > EMA200)
            if williams_r[i] < -80 and ema50_1d_aligned[i] > ema200_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND 1d bearish trend (EMA50 < EMA200)
            elif williams_r[i] > -20 and ema50_1d_aligned[i] < ema200_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses back through -50 mean level
            exit_long = (position == 1 and williams_r[i] > -50)
            exit_short = (position == -1 and williams_r[i] < -50)
            
            if exit_long or exit_short:
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

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter
# - Long when Williams %R(14) < -80 (oversold) AND 1d EMA50 > EMA200 (bullish trend)
# - Short when Williams %R(14) > -20 (overbought) AND 1d EMA50 < EMA200 (bearish trend)
# - Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams %R captures short-term reversals; 1d EMA filter ensures we trade with the higher timeframe trend
# - Mean reversion works well in ranging markets which dominate BTC/ETH price action
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_williamsr_meanreversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute 6h Williams %R (14-period)
    def highest_high(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def lowest_low(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    hh_6h = highest_high(high, 14)
    ll_6h = lowest_low(low, 14)
    williams_r = np.full_like(close, np.nan, dtype=float)
    for i in range(13, len(close)):
        if hh_6h[i] > ll_6h[i]:
            williams_r[i] = (hh_6h[i] - close[i]) / (hh_6h[i] - ll_6h[i]) * -100
        else:
            williams_r[i] = -50.0
    
    # Pre-compute 1d EMA50 and EMA200
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND 1d bullish trend (EMA50 > EMA200)
            if williams_r[i] < -80 and ema50_1d_aligned[i] > ema200_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND 1d bearish trend (EMA50 < EMA200)
            elif williams_r[i] > -20 and ema50_1d_aligned[i] < ema200_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses back through -50 mean level
            exit_long = (position == 1 and williams_r[i] > -50)
            exit_short = (position == -1 and williams_r[i] < -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals