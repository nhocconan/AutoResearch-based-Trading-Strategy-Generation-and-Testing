#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX regime filter
# - Long when price breaks above 20-period Donchian high AND ADX > 25 (trending) AND volume > 1.5x 20-period volume SMA
# - Short when price breaks below 20-period Donchian low AND ADX > 25 (trending) AND volume > 1.5x 20-period volume SMA
# - Exit: Donchian midpoint reversion
# - Position sizing: 0.30 discrete level
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - ADX filter ensures we only trade in trending markets, reducing whipsaw in ranging conditions
# - Volume confirmation adds conviction to breakouts
# - Donchian provides objective breakout levels with clear exit at midpoint

name = "4h_1d_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 20-period Donchian channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX(14) for regime filtering
    # ADX calculation requires +DM, -DM, and TR
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    
    # Pad arrays to match original length
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    tr = np.concatenate([[np.nan], tr])
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    plus_di_14 = 100 * wilders_smoothing(plus_dm, atr_period) / atr
    minus_di_14 = 100 * wilders_smoothing(minus_dm, atr_period) / atr
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = wilders_smoothing(dx, atr_period)
    
    # Track entry extreme for exit logic
    entry_price = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(volume_sma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # ADX trend filter: ADX > 25 indicates trending market
        trend_filter = adx[i] > 25
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and trend_filter:
                position = 1
                signals[i] = 0.30
                entry_price[i] = close[i]
            elif breakout_down and vol_confirm and trend_filter:
                position = -1
                signals[i] = -0.30
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on Donchian midpoint reversion
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.30
        else:  # position == -1 (Short position) - look for exit
            # Exit on Donchian midpoint reversion
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.30
    
    return signals