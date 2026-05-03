#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
# In bull regime (price > 12h EMA50), go long on breakout above upper Donchian with volume spike.
# In bear regime (price < 12h EMA50), go short on breakdown below lower Donchian with volume spike.
# Uses Donchian channels from 4h for structure, 12h EMA50 for regime filter, and 4h volume spike for confirmation.
# Designed for 75-200 total trades over 4 years (19-50/year) on BTC/ETH. SOL as secondary.
# ATR-based stoploss: exit when price moves against position by 2.5 * ATR(14).

name = "4h_Donchian20_12hEMA50_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        upper_dc = highest_high[i]
        lower_dc = lowest_low[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(upper_dc) or np.isnan(lower_dc) or np.isnan(ema_trend) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 12h EMA50, bear if close < 12h EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: breakout above upper Donchian with volume spike
            long_entry = (close_val > upper_dc) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: breakdown below lower Donchian with volume spike
            short_entry = (close_val < lower_dc) and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss, breakdown below lower DC, or regime change to bear
            stoploss_hit = close_val < entry_price - 2.5 * atr_val
            breakout_fail = close_val < lower_dc
            regime_change = close_val < ema_trend
            if stoploss_hit or breakout_fail or regime_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: stoploss, breakout above upper DC, or regime change to bull
            stoploss_hit = close_val > entry_price + 2.5 * atr_val
            breakout_fail = close_val > upper_dc
            regime_change = close_val > ema_trend
            if stoploss_hit or breakout_fail or regime_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals