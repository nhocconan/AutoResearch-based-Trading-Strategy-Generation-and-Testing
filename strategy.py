#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d trend filter + ATR stoploss
# - Long when price breaks above Donchian(20) upper band with volume > 1.8x 20-bar average AND 1d close > 1d EMA50
# - Short when price breaks below Donchian(20) lower band with volume > 1.8x 20-bar average AND 1d close < 1d EMA50
# - Exit when price retreats to Donchian midpoint OR ATR-based stoploss hit
# - Uses 1d trend filter to avoid counter-trend trades and ATR stoploss for risk control
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)
# - Focus on BTC/ETH; SOL-only strategies are low value

name = "4h_1d_donchian_breakout_volume_trend_atrstop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute volume filter for exit: < 0.8x average volume (loss of momentum)
    vol_weak = prices['volume'] < (0.8 * volume_20_avg)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute Donchian channels (20-period)
    donchian_upper = prices['high'].rolling(window=20, min_periods=20).max().values
    donchian_lower = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
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
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(atr[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or 
            np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Donchian upper with volume spike AND 1d uptrend
            if (prices['close'].iloc[i] > donchian_upper[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                signals[i] = 0.25
            # Short breakdown: price < Donchian lower with volume spike AND 1d downtrend
            elif (prices['close'].iloc[i] < donchian_lower[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retreats to Donchian midpoint
            # 2. Volume drops below 0.8x average (loss of momentum)
            # 3. ATR-based stoploss hit
            exit_signal = False
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < donchian_mid[i] or 
                    vol_weak.iloc[i] or
                    prices['close'].iloc[i] < entry_price - 2.5 * atr[i]):
                    exit_signal = True
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > donchian_mid[i] or 
                    vol_weak.iloc[i] or
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