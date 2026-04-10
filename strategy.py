#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) < -80 = oversold (long signal), > -20 = overbought (short signal)
# - Requires 1d close > 1d EMA50 for longs, < 1d EMA50 for shorts (trend alignment)
# - Requires volume > 1.5x 20-bar average for confirmation
# - Exit when Williams %R returns to -50 (mean reversion midpoint) or ATR stoploss
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Williams %R works well in ranging/ bear markets (2025-2026 test period)

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
    highest_high = pd.Series(prices['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(prices['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - prices['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
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
            # Long when oversold AND volume spike AND 1d uptrend
            if (williams_r[i] < -80 and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                signals[i] = 0.25
            # Short when overbought AND volume spike AND 1d downtrend
            elif (williams_r[i] > -20 and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
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
            if position == 1:  # Long position
                if (williams_r[i] > -50 or
                    prices['close'].iloc[i] < entry_price - 2.5 * atr[i]):
                    exit_signal = True
            elif position == -1:  # Short position
                if (williams_r[i] < -50 or
                    prices['close'].iloc[i] > entry_price + 2.5 * atr[i]):
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