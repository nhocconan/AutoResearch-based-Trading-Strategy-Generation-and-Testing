#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high with volume spike AND 1d EMA34 uptrend
# Short when price breaks below Donchian(20) low with volume spike AND 1d EMA34 downtrend
# Exit on opposite Donchian breakout or trend reversal
# Uses proven Donchian breakout structure with volume and trend filters to reduce false signals
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag while capturing strong trends

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels from 4h data
    high_4h = high
    low_4h = low
    
    # Donchian high: highest high over last 20 periods
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over last 20 periods
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_dc_high = donchian_high[i]
        curr_dc_low = donchian_low[i]
        curr_ema34 = ema34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        is_bullish_regime = curr_close > curr_ema34
        is_bearish_regime = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation to avoid false breakouts
            if curr_volume_confirm:
                # Bullish entry: price breaks above Donchian high with volume AND bullish regime
                if curr_high > curr_dc_high and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian low with volume AND bearish regime
                elif curr_low < curr_dc_low and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when price falls below Donchian low or regime changes
            if curr_low < curr_dc_low or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price rises above Donchian high or regime changes
            if curr_high > curr_dc_high or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals