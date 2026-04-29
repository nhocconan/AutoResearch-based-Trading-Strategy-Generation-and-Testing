#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume spike
# Donchian breakouts capture momentum; 1d EMA50 filters for primary trend direction
# Volume spike confirms institutional participation
# Long when price breaks above 20-period Donchian high AND price > 1d EMA50 AND volume spike
# Short when price breaks below 20-period Donchian low AND price < 1d EMA50 AND volume spike
# Exit when price returns to Donchian midpoint (mean reversion within channel)
# Uses 6h timeframe targeting 50-150 total trades (12-37/year) to minimize fee drag
# Works in bull/bear markets by following 1d trend while capturing 6h momentum

name = "6h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian Channel (20-period) on 6h data
    # Upper band: highest high over last 20 periods
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: volume > 2.0x 20-period average (20*6h = 5 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        curr_donchian_middle = donchian_middle[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA50, bearish if price < 1d EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price breaks above Donchian upper AND bullish regime
                if curr_high > curr_donchian_upper and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian lower AND bearish regime
                elif curr_low < curr_donchian_lower and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when price returns to Donchian midpoint (mean reversion)
            if curr_close <= curr_donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when price returns to Donchian midpoint (mean reversion)
            if curr_close >= curr_donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals