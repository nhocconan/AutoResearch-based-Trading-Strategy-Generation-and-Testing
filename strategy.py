#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR stoploss
# - Long: Price breaks above Donchian(20) high + volume > 1.5x 20-period average
# - Short: Price breaks below Donchian(20) low + volume > 1.5x 20-period average
# - Exit: ATR-based stop (2.0 ATR) or opposite Donchian breakout
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits
# - Donchian channels capture momentum breakouts effectively in trending markets
# - Volume confirmation filters false breakouts
# - ATR stoploss manages risk during adverse moves

name = "4h_12h_donchian_volume_atrstop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute 12h volume SMA(20)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute Donchian(20) on primary timeframe (4h)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR(14) for stoploss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_high = close_price > donchian_high[i-1]  # Using previous bar's high
        breakout_low = close_price < donchian_low[i-1]   # Using previous bar's low
        
        # ATR-based stoploss levels
        if position == 1:
            stop_price = entry_price - 2.0 * atr_14[i]
        elif position == -1:
            stop_price = entry_price + 2.0 * atr_14[i]
        else:
            stop_price = 0.0
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long entry: price breaks above Donchian high with volume confirmation
        if breakout_high and vol_confirm and position != 1:
            enter_long = True
        
        # Short entry: price breaks below Donchian low with volume confirmation
        if breakout_low and vol_confirm and position != -1:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on ATR stoploss or opposite breakout
            exit_long = (close_price <= stop_price) or breakout_low
        elif position == -1:
            # Exit short on ATR stoploss or opposite breakout
            exit_short = (close_price >= stop_price) or breakout_high
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals