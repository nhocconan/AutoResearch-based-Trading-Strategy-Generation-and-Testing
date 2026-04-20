#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data once for ATR, volume, and close
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Daily ATR (14) - true range
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Daily volume average (20)
    vol_ma_daily = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_daily_aligned[i]) or np.isnan(vol_ma_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_daily = atr_daily_aligned[i]
        vol_ma_daily = vol_ma_daily_aligned[i]
        vol_current = volume[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter_ok = atr_daily > 0
        
        # Volume filter: current volume > 2x daily average
        vol_ok = vol_current > 2.0 * vol_ma_daily
        
        # Price range filter: avoid extreme ranges (> 4x ATR)
        price_range = high[i] - low[i]
        range_filter_ok = price_range < 4.0 * atr_daily
        
        if position == 0:
            # Long: high volume + normal volatility + price near daily low
            if vol_ok and vol_filter_ok and range_filter_ok:
                # Buy when price is in lower 20% of daily range AND volume spike
                daily_low = low_daily[i]  # This is the daily low for the current day
                daily_high = high_daily[i]
                if daily_high > daily_low:
                    price_position = (price - daily_low) / (daily_high - daily_low)
                    if price_position < 0.2:  # Lower 20% of daily range
                        signals[i] = 0.25
                        position = 1
            # Short: high volume + normal volatility + price near daily high
            elif vol_ok and vol_filter_ok and range_filter_ok:
                # Sell when price is in upper 20% of daily range AND volume spike
                daily_low = low_daily[i]
                daily_high = high_daily[i]
                if daily_high > daily_low:
                    price_position = (price - daily_low) / (daily_high - daily_low)
                    if price_position > 0.8:  # Upper 20% of daily range
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Long exit: price reaches middle of daily range OR volatility drops
            daily_low = low_daily[i]
            daily_high = high_daily[i]
            if daily_high > daily_low:
                price_position = (price - daily_low) / (daily_high - daily_low)
                if price_position > 0.5:  # Above midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short exit: price reaches middle of daily range OR volatility drops
            daily_low = low_daily[i]
            daily_high = high_daily[i]
            if daily_high > daily_low:
                price_position = (price - daily_low) / (daily_high - daily_low)
                if price_position < 0.5:  # Below midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_1d_VolumeSpike_RangePosition_v1"
timeframe = "4h"
leverage = 1.0