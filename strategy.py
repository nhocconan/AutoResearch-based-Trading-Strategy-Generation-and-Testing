#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation + ATR stop
# - Long when price breaks above Donchian(20) high with volume > 1.5x average AND 12h close > EMA34
# - Short when price breaks below Donchian(20) low with volume > 1.5x average AND 12h close < EMA34
# - Exit when price retests Donchian(20) midline OR ATR-based stoploss (2*ATR from entry)
# - 12h EMA34 trend filter ensures alignment with medium-term trend
# - Volume confirmation prevents false breakouts
# - Targets 20-35 trades/year (80-140 total over 4 years) to avoid fee drag
# - Donchian breakouts work in both trending and ranging markets with proper filters

name = "4h_12h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian high and low (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_at_entry = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema34_12h_aligned[i]) or
            np.isnan(volume_20_avg[i]) or np.isnan(atr[i])):
            # Hold current position or flatten if invalid
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Donchian high with volume spike AND 12h uptrend
            if (prices['close'].iloc[i] > donchian_high[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema34_12h_aligned[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                atr_at_entry = atr[i]
                signals[i] = 0.25
            # Short breakdown: price < Donchian low with volume spike AND 12h downtrend
            elif (prices['close'].iloc[i] < donchian_low[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema34_12h_aligned[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                atr_at_entry = atr[i]
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests Donchian midline (mean reversion signal)
            # 2. ATR-based stoploss (2*ATR from entry)
            if position == 1:  # Long position
                midline_touch = prices['close'].iloc[i] < donchian_mid[i]
                stoploss_hit = prices['close'].iloc[i] < (entry_price - 2.0 * atr_at_entry)
                if midline_touch or stoploss_hit:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                midline_touch = prices['close'].iloc[i] > donchian_mid[i]
                stoploss_hit = prices['close'].iloc[i] > (entry_price + 2.0 * atr_at_entry)
                if midline_touch or stoploss_hit:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals