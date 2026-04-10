#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w EMA50 trend filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 2.0x 20-bar average AND 1w close > 1w EMA50
# - Short when price breaks below Donchian(20) low AND 1d volume > 2.0x 20-bar average AND 1w close < 1w EMA50
# - Exit when price crosses Donchian(20) midline (10-bar average of high/low) or ATR-based stoploss
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25-35 trades/year (100-140 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong momentum moves in both bull and bear markets
# - Volume confirmation ensures breakouts have conviction
# - 1w EMA50 filter ensures we trade with higher timeframe momentum

name = "4h_1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels on 4h data
    donchian_period = 20
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian high and low
    donchian_high = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute ATR for stoploss
    atr_period = 14
    tr1 = pd.Series(high_4h - low_4h).values
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1))).values
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1))).values
    tr2[0] = tr1[0]  # First bar has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_multiplier = 2.5
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high with volume spike and 1w uptrend
            if (close_4h[i] > donchian_high[i] and 
                vol_spike.iloc[i] and 
                close_4h[i] > ema_50_1w_aligned[i]):
                position = 1
                entry_price = close_4h[i]
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low with volume spike and 1w downtrend
            elif (close_4h[i] < donchian_low[i] and 
                  vol_spike.iloc[i] and 
                  close_4h[i] < ema_50_1w_aligned[i]):
                position = -1
                entry_price = close_4h[i]
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian midline
            # 2. ATR-based stoploss
            if position == 1:  # Long position
                midline_cross = close_4h[i] < donchian_mid[i]
                stoploss_hit = close_4h[i] < entry_price - (atr_multiplier * atr[i])
                if midline_cross or stoploss_hit:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                midline_cross = close_4h[i] > donchian_mid[i]
                stoploss_hit = close_4h[i] > entry_price + (atr_multiplier * atr[i])
                if midline_cross or stoploss_hit:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals