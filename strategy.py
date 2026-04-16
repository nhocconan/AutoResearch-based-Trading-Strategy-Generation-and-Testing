#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and 6h volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend bias) AND 6h volume > 2.0x 20-period average.
# Short when Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend bias) AND 6h volume > 2.0x 20-period average.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position size 0.25. Williams %R identifies exhaustion points, 1d EMA34 provides trend bias to avoid counter-trend traps,
# volume confirmation ensures participation. Designed for 60-120 trades over 4 years (15-30/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # === 6h Indicators: Williams %R(14) ===
    # Highest high over 14 periods
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    # Lowest low over 14 periods
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close_6h) / (highest_high - lowest_low)) * -100, -50)
    
    # Align Williams %R to 6h timeframe (no additional delay needed as it's not a lagging confirmation indicator)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d data once before loop for EMA34 filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA(34) for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 6h data for volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        wr_val = williams_r_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 2.0x 20-period average (using same timeframe volume)
        vol_filter = vol > 2.0 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50 (exiting oversold territory)
            if wr_val > -50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50 (exiting overbought territory)
            if wr_val < -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend bias) AND volume confirmation
            if wr_val < -80 and price > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend bias) AND volume confirmation
            elif wr_val > -20 and price < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR_MeanReversion_1dEMA34_6hVolumeSpike_V1"
timeframe = "6h"
leverage = 1.0