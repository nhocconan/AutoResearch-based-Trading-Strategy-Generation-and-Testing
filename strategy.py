#12h_Keltner_Channel_Breakout_1dTrend_Filter
# Hypothesis: 12h Keltner Channel breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above upper KC (EMA20 + 2*ATR) with 1d uptrend and volume spike
# Short when price breaks below lower KC (EMA20 - 2*ATR) with 1d downtrend and volume spike
# Keltner Channels adapt to volatility, reducing false breakouts in low volatility periods
# Trend filter ensures trades align with higher timeframe momentum
# Volume spike confirms institutional participation
# Designed for 12h timeframe targeting 15-25 trades/year per symbol
# Works in both bull (trend continuation) and bear (trend reversal) markets

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
    
    # Load 1d data for trend filter and ATR (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR(14) for Keltner Channel width
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # EMA(20) for Keltner Channel middle (on 12h data)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bounds
    kc_upper = ema_20 + 2 * atr_14_aligned
    kc_lower = ema_20 - 2 * atr_14_aligned
    
    # Volume spike filter (20-period on 12h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper KC + 1d uptrend + volume spike
            if (close[i] > kc_upper[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower KC + 1d downtrend + volume spike
            elif (close[i] < kc_lower[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to EMA20 or trend reversal
            if position == 1:
                # Exit on price below EMA20 or trend reversal
                if (close[i] < ema_20[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above EMA20 or trend reversal
                if (close[i] > ema_20[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Keltner_Channel_Breakout_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0