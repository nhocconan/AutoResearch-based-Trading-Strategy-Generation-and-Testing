#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h trend (EMA34) and 1d momentum (RSI14) for direction,
# with 1h Donchian breakout (20) + volume confirmation for entry.
# Trend-following in bull/bear via 4h EMA34 filter, momentum confirmation via 1d RSI>50 for long/<50 for short.
# Volume spike (>1.5x 20MA) ensures institutional participation. Session filter (08-20 UTC) reduces noise.
# Target: 15-30 trades/year via tight multi-condition entry.
name = "1h_4hEMA34_1dRSI14_Donchian20_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA34 trend (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for RSI14 momentum (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # RSI(14) calculation
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi_14_1d.fillna(50).values  # neutral when undefined
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Get 1h data for Donchian20 breakout and volume filter
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)  # volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend (price > EMA34) AND 1d bullish momentum (RSI>50) 
            #        AND breaks 1h Donchian high with volume spike
            if (close[i] > ema_34_4h_aligned[i] and 
                rsi_14_1d_aligned[i] > 50 and 
                close[i] > high_20[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend (price < EMA34) AND 1d bearish momentum (RSI<50) 
            #        AND breaks 1h Donchian low with volume spike
            elif (close[i] < ema_34_4h_aligned[i] and 
                  rsi_14_1d_aligned[i] < 50 and 
                  close[i] < low_20[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if 4h trend breaks or 1h Donchian low broken
            if close[i] < ema_34_4h_aligned[i] or close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if 4h trend breaks or 1h Donchian high broken
            if close[i] > ema_34_4h_aligned[i] or close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals