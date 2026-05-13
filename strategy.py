#!/usr/bin/env python3
# 6h_Aggressive_Accumulator_Detection
# Hypothesis: Detects accumulation/distribution phases using volume-price divergence and
# breaks of short-term structure with multi-timeframe confirmation. Works in bull markets
# by catching early breakouts from accumulation, and in bear markets by catching
# distribution breakdowns before major moves. Uses 60-period volume-weighted RSI to filter
# choppy markets and avoid false breakouts. Target: 20-40 trades/year.

name = "6h_Aggressive_Accumulator_Detection"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: using align_ltf_to_htf is incorrect, should be align_htf_to_ltf
# Correction: will use align_htf_to_ltf as per rules

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 60-period volume-weighted RSI (proxy for institutional flow)
    # Calculate typical price and money flow
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    # Positive and negative money flow
    price_change = np.diff(typical_price, prepend=typical_price[0])
    pos_mf = np.where(price_change > 0, money_flow, 0)
    neg_mf = np.where(price_change < 0, money_flow, 0)
    
    # 14-period money flow ratio
    pos_mf_sum = pd.Series(pos_mf).rolling(window=14, min_periods=14).sum()
    neg_mf_sum = pd.Series(neg_mf).rolling(window=14, min_periods=14).sum()
    mf_ratio = np.where(neg_mf_sum != 0, pos_mf_sum / neg_mf_sum, 100)
    vw_rsi = 100 - (100 / (1 + mf_ratio))
    
    # Get 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w structure (Higher Highs/Lower Lows)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    # Weekly pivot points (simplified: using weekly high/low)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume spike detector (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(vw_rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]):
            signals[i] = 0.0
            continue
            
        # LONG: Price above weekly low with bullish momentum and volume spike
        # But only if not in overbought territory (vw_rsi < 70)
        if (close[i] > weekly_low_aligned[i] and 
            vw_rsi[i] > 50 and vw_rsi[i] < 70 and 
            volume_spike[i] and 
            close[i] > ema_50_1d_aligned[i]):
            signals[i] = 0.25
            position = 1
        # SHORT: Price below weekly high with bearish momentum and volume spike
        # But only if not in oversold territory (vw_rsi > 30)
        elif (close[i] < weekly_high_aligned[i] and 
              vw_rsi[i] < 50 and vw_rsi[i] > 30 and 
              volume_spike[i] and 
              close[i] < ema_50_1d_aligned[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Exit conditions: loss of momentum or contrary volume spike
            if position == 1:
                if vw_rsi[i] < 40 or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if vw_rsi[i] > 60 or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
# 6h_Aggressive_Accumulator_Detection
# Hypothesis: Detects accumulation/distribution phases using volume-price divergence and
# breaks of short-term structure with multi-timeframe confirmation. Works in bull markets
# by catching early breakouts from accumulation, and in bear markets by catching
# distribution breakdowns before major moves. Uses 60-period volume-weighted RSI to filter
# choppy markets and avoid false breakouts. Target: 20-40 trades/year.

name = "6h_Aggressive_Accumulator_Detection"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 60-period volume-weighted RSI (proxy for institutional flow)
    # Calculate typical price and money flow
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    # Positive and negative money flow
    price_change = np.diff(typical_price, prepend=typical_price[0])
    pos_mf = np.where(price_change > 0, money_flow, 0)
    neg_mf = np.where(price_change < 0, money_flow, 0)
    
    # 14-period money flow ratio
    pos_mf_sum = pd.Series(pos_mf).rolling(window=14, min_periods=14).sum()
    neg_mf_sum = pd.Series(neg_mf).rolling(window=14, min_periods=14).sum()
    mf_ratio = np.where(neg_mf_sum != 0, pos_mf_sum / neg_mf_sum, 100)
    vw_rsi = 100 - (100 / (1 + mf_ratio))
    
    # Get 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w structure (Higher Highs/Lower Lows)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    # Weekly pivot points (simplified: using weekly high/low)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume spike detector (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(vw_rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]):
            signals[i] = 0.0
            continue
            
        # LONG: Price above weekly low with bullish momentum and volume spike
        # But only if not in overbought territory (vw_rsi < 70)
        if (close[i] > weekly_low_aligned[i] and 
            vw_rsi[i] > 50 and vw_rsi[i] < 70 and 
            volume_spike[i] and 
            close[i] > ema_50_1d_aligned[i]):
            signals[i] = 0.25
            position = 1
        # SHORT: Price below weekly high with bearish momentum and volume spike
        # But only if not in oversold territory (vw_rsi > 30)
        elif (close[i] < weekly_high_aligned[i] and 
              vw_rsi[i] < 50 and vw_rsi[i] > 30 and 
              volume_spike[i] and 
              close[i] < ema_50_1d_aligned[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Exit conditions: loss of momentum or contrary volume spike
            if position == 1:
                if vw_rsi[i] < 40 or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if vw_rsi[i] > 60 or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals