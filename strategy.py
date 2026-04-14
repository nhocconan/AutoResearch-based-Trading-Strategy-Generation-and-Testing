#43075
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d volatility regime
# Long when price touches 1h lower Bollinger Band AND 4h EMA(50) is rising AND 1d volatility is low
# Short when price touches 1h upper Bollinger Band AND 4h EMA(50) is falling AND 1d volatility is low
# Exit when price crosses Bollinger middle
# Bollinger Bands capture mean reversion; 4h EMA ensures higher timeframe trend alignment; 
# Low volatility regime avoids false signals during choppy periods
# Designed for 1h timeframe with 4h/1d filters to reduce noise and trade frequency
# Target: 60-150 total trades over 4 years (15-37/year) to manage fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Load 1d data ONCE before loop for volatility regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20
    
    # Calculate 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.inf
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Calculate 1d ATR percentile rank (252-period ~ 1 year)
    atr_percentile = pd.Series(atr_14).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    )
    
    # Align 4h EMA and 1d ATR percentile to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h.values)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or 
            np.isnan(middle_bb[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_val = ema_50_4h_aligned[i]
        ema_prev = ema_50_4h_aligned[i-1]
        atr_percentile_val = atr_percentile_aligned[i]
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Only trade in low volatility regime (ATR percentile < 40%)
        volatility_filter = atr_percentile_val < 0.4
        
        if position == 0:
            # Long setup: price touches lower Bollinger Band AND 4h EMA rising AND low volatility
            if (low_val <= lower_bb[i] and ema_val > ema_prev and volatility_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price touches upper Bollinger Band AND 4h EMA falling AND low volatility
            elif (high_val >= upper_bb[i] and ema_val < ema_prev and volatility_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above Bollinger middle
            if close_val > middle_bb[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below Bollinger middle
            if close_val < middle_bb[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_Bollinger_4hEMA_1dVolatility"
timeframe = "1h"
leverage = 1.0