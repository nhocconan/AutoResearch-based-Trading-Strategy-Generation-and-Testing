#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) measures overbought/oversold levels (-100 to 0)
# - Long when Williams %R crosses above -80 from below AND price > 1d EMA50 (uptrend) AND volume > 1.5x average
# - Short when Williams %R crosses below -20 from above AND price < 1d EMA50 (downtrend) AND volume > 1.5x average
# - Exit when Williams %R returns to -50 (mean reversion midpoint) OR ATR stoploss hit
# - Uses 1d trend filter to avoid counter-trend trades in ranging/choppy markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)
# - Williams %R is effective in both bull and bear markets for mean reversion swings

name = "6h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14)
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # track entry price for stoploss
    
    # Pre-compute aligned 1d data
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(atr[i]) or 
            np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or 
            np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Williams %R conditions for entry
            wr_current = williams_r[i]
            wr_previous = williams_r[i-1]
            
            # Long when Williams %R crosses above -80 from below (oversold bounce)
            long_condition = (wr_previous <= -80 and wr_current > -80 and 
                            prices['close'].iloc[i] > ema50_1d_aligned[i] and
                            vol_spike.iloc[i])
            
            # Short when Williams %R crosses below -20 from above (overbought rejection)
            short_condition = (wr_previous >= -20 and wr_current < -20 and 
                             prices['close'].iloc[i] < ema50_1d_aligned[i] and
                             vol_spike.iloc[i])
            
            if long_condition:
                position = 1
                entry_price = prices['close'].iloc[i]
                signals[i] = 0.25
            elif short_condition:
                position = -1
                entry_price = prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to -50 (mean reversion midpoint)
            # 2. ATR-based stoploss hit
            exit_signal = False
            wr_current = williams_r[i]
            
            if position == 1:  # Long position
                if (wr_current >= -50 or  # Mean reversion exit
                    prices['close'].iloc[i] < entry_price - 2.5 * atr[i]):  # ATR stoploss
                    exit_signal = True
            elif position == -1:  # Short position
                if (wr_current <= -50 or  # Mean reversion exit
                    prices['close'].iloc[i] > entry_price + 2.5 * atr[i]):  # ATR stoploss
                    exit_signal = True
            
            if exit_signal:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals